"""gat_agent_tool.core — the GAT wrapped as a stateful scorer.

`GatScorerTool` loads a trained GAT checkpoint once and exposes a small, clean
scoring API that's safe to call many times:

    >>> from gat_agent_tool import GatScorerTool
    >>> scorer = GatScorerTool("path/to/gat_model.pt")
    >>> scorer.score("(lab_RDW_mean + lab_RBC_max) / lab_WBC_mean")
    0.6832...
    >>> scorer.score_batch(["lab_A + lab_B", "bogus + 1"])
    [0.61, None]
    >>> scorer.feature_names[:3]
    ['lab_RDW_mean', 'lab_RBC_max', 'lab_WBC_mean']

This is the reusable core that all three consumer paths (local-HF loop, MCP
server, OpenAI function-calling) share. Internally it reuses the GAT model
classes and graph-conversion code from the main repo's `Code/` directory; the
package's ``__init__.py`` adds that directory to sys.path so the imports just
work.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

# These come from the main BD_Paper repo's Code/ — the package __init__ makes
# them importable. If this tool is ever extracted and used on its own, the user
# must supply gat_model.py + expr_graph_utils.py or set GAT_AGENT_TOOL_CODE_DIR.
from expr_graph_utils import string_to_data_obj  # type: ignore
from gat_model import (  # type: ignore
    ACTIVATION_MAP,
    NUM_EDGE_TYPES,
    AdvancedGAT,
    EfficientGAT,
    TreeAwareGAT,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _inverse_auc_transform(raw_score: float, transform: Dict) -> float:
    """Invert the AUC transformation stored in the GAT checkpoint.

    Mirrors the inverse in evaluate_unified.py:3426-3445 so scores from this
    tool are numerically identical to scores from the main pipeline's
    ``score_expressions_with_gat`` function.
    """
    ttype = transform.get("type")
    if ttype == "log_scale":
        epsilon = float(transform.get("epsilon", 1e-10))
        log_shift = float(transform.get("log_shift", 15))
        log_scale = float(transform.get("log_scale", 20))
        return float(np.exp(raw_score / log_scale - log_shift) - epsilon + 0.5)
    if ttype in {"linear_0_1", "linear"}:
        auc_min = float(transform.get("min", 0.5))
        auc_max = float(transform.get("max", 1.0))
        return raw_score * (auc_max - auc_min) + auc_min
    if ttype == "power":
        auc_min = float(transform.get("min", 0.5))
        auc_max = float(transform.get("max", 1.0))
        power = float(transform.get("power", 1.0))
        norm = max(0.0, raw_score)
        if power != 0.0:
            norm = norm ** (1.0 / power)
        return norm * (auc_max - auc_min) + auc_min
    return float(raw_score)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelInfo:
    """Descriptive metadata about a loaded GAT checkpoint. Returned by
    :meth:`GatScorerTool.info` and exposed via the MCP ``get_feature_list`` tool.
    """
    model_type: str
    num_features: int
    num_operators: int
    max_depth: int
    graph_format_version: int
    has_auc_transform: bool
    checkpoint_path: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class GatScorerTool:
    """A trained GAT, loaded once, queryable as an expression-scoring tool.

    Parameters
    ----------
    checkpoint_path : str
        Path to a checkpoint saved by the main pipeline's training scripts.
    device : str, default "auto"
        "auto" picks cuda if available, else cpu. Pass an explicit torch device
        string (e.g. "cuda:0", "cpu") to force a placement.
    batch_size : int, default 128
        Upper bound on graphs per forward pass in :meth:`score_batch`.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        batch_size: int = 128,
    ) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GAT checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.batch_size = int(batch_size)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Metadata the LLM may want to introspect (features, operators, etc.).
        gat_cfg = ckpt["best_config"]
        self._op_map: Dict[str, int] = ckpt["op_mapping"]
        self._feat_map: Dict[str, int] = ckpt["feature_mapping"]
        num_ops = int(ckpt["num_operators"])
        num_feats = int(ckpt["num_features"])
        self._graph_format_version = int(ckpt.get("graph_format_version", 1))
        self._num_edge_types = int(ckpt.get("num_edge_types", NUM_EDGE_TYPES))
        self._max_depth = int(ckpt.get("max_depth_used", gat_cfg.get("max_depth_used", 32)))

        auc_transform = ckpt.get("auc_transform") or ckpt.get("ig_transform")
        if isinstance(auc_transform, dict) and auc_transform.get("type"):
            self._auc_transform: Optional[Dict] = auc_transform
            logger.info(
                "GatScorerTool: checkpoint uses AUC transformation type=%s",
                auc_transform.get("type"),
            )
        else:
            self._auc_transform = None

        emb_dim_type = int(gat_cfg["embedding_dim_type"])
        emb_dim_id = int(gat_cfg["embedding_dim_id"])
        emb_dim_position = int(gat_cfg.get("embedding_dim_position", 16))
        act_fn_name = gat_cfg.get("activation_fn_name", "elu")
        act_fn = ACTIVATION_MAP.get(act_fn_name, torch.nn.ELU)
        model_type = ckpt.get("model_type", "EfficientGAT")
        self._model_type = model_type

        # Dispatch to the right GAT class. Same logic as
        # evaluate_unified.py:3472-3523 in the main pipeline.
        if model_type == "TreeAwareGAT":
            emb_dim_depth = int(gat_cfg.get("embedding_dim_depth", 16))
            emb_dim_edge = int(gat_cfg.get("embedding_dim_edge", 8))
            self.model = TreeAwareGAT(
                num_operators=num_ops,
                num_features=num_feats,
                embedding_dim_type=emb_dim_type,
                embedding_dim_id=emb_dim_id,
                embedding_dim_position=emb_dim_position,
                embedding_dim_depth=emb_dim_depth,
                edge_embedding_dim=emb_dim_edge,
                hidden_channels_gat=int(gat_cfg["hidden_channels_gat"]),
                out_channels_final=1,
                num_layers_gat=int(gat_cfg["num_layers_gat"]),
                heads_gat=int(gat_cfg["heads_gat"]),
                dropout_rate=float(gat_cfg["dropout_rate"]),
                pooling_type_gat=gat_cfg["pooling_type_gat"],
                activation_fn_gat=act_fn,
                num_edge_types=self._num_edge_types,
                max_depth=self._max_depth,
            ).to(self.device)
        elif model_type == "AdvancedGAT":
            self.model = AdvancedGAT(
                num_operators=num_ops,
                num_features=num_feats,
                embedding_dim_type=emb_dim_type,
                embedding_dim_id=emb_dim_id,
                hidden_channels_gat=int(gat_cfg["hidden_channels_gat"]),
                out_channels_final=1,
                num_layers_gat=int(gat_cfg["num_layers_gat"]),
                heads_gat=int(gat_cfg["heads_gat"]),
                dropout_rate=float(gat_cfg["dropout_rate"]),
                pooling_type_gat=gat_cfg["pooling_type_gat"],
                activation_fn_gat=act_fn,
                embedding_dim_position=emb_dim_position,
            ).to(self.device)
        else:
            self.model = EfficientGAT(
                num_operators=num_ops,
                num_features=num_feats,
                embedding_dim_type=emb_dim_type,
                embedding_dim_id=emb_dim_id,
                hidden_channels_gat=int(gat_cfg["hidden_channels_gat"]),
                out_channels_final=1,
                num_layers_gat=int(gat_cfg["num_layers_gat"]),
                heads_gat=int(gat_cfg["heads_gat"]),
                dropout_rate=float(gat_cfg["dropout_rate"]),
                pooling_type_gat=gat_cfg["pooling_type_gat"],
                activation_fn_gat=act_fn,
                embedding_dim_position=emb_dim_position,
            ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info(
            "GatScorerTool loaded %s: %d ops, %d features, device=%s",
            model_type, num_ops, num_feats, self.device,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def feature_names(self) -> List[str]:
        """Feature names the GAT was trained on (case-sensitive)."""
        return list(self._feat_map.keys())

    @property
    def operators(self) -> List[str]:
        """Arithmetic operator tokens the GAT recognizes (currently ``+ - * /``)."""
        return list(self._op_map.keys())

    @property
    def max_depth(self) -> int:
        """Maximum tree depth seen during GAT training. Deeper expressions
        generalise poorly — the LLM prompt should keep under this limit."""
        return self._max_depth

    @property
    def model_type(self) -> str:
        """Which GAT variant is loaded: EfficientGAT, AdvancedGAT, or TreeAwareGAT."""
        return self._model_type

    def info(self) -> ModelInfo:
        """Return all metadata in one dataclass, convenient for JSON serialization."""
        return ModelInfo(
            model_type=self._model_type,
            num_features=len(self._feat_map),
            num_operators=len(self._op_map),
            max_depth=self._max_depth,
            graph_format_version=self._graph_format_version,
            has_auc_transform=self._auc_transform is not None,
            checkpoint_path=self.checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, expression: str) -> Optional[float]:
        """Score a single expression. Returns a predicted AUC in ``[0.5, 1.0]``,
        or ``None`` if the expression cannot be parsed into a graph (e.g.
        unknown feature, unbalanced parens, contains a numeric literal)."""
        out = self.score_batch([expression])
        return out[0] if out else None

    def score_batch(self, expressions: List[str]) -> List[Optional[float]]:
        """Score a list of expressions. Output order matches input order;
        entries that fail graph conversion come back as ``None``."""
        if not expressions:
            return []

        # 1) Convert each expression to a PyG Data object; track which survived.
        data_objs = []
        valid_positions: List[int] = []
        for idx, expr in enumerate(expressions):
            if not isinstance(expr, str) or not expr.strip():
                continue
            data_obj = string_to_data_obj(
                expr,
                self._op_map,
                self._feat_map,
                graph_format_version=self._graph_format_version,
            )
            if data_obj is None or data_obj.x is None or data_obj.x.shape[0] == 0:
                continue
            data_objs.append(data_obj)
            valid_positions.append(idx)

        results: List[Optional[float]] = [None] * len(expressions)
        if not data_objs:
            return results

        # 2) Batch forward pass.
        loader = PyGDataLoader(
            data_objs,
            batch_size=min(self.batch_size, len(data_objs)) or 1,
            shuffle=False,
        )
        scored: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self.model(batch)
                for i in range(preds.size(0)):
                    raw_score = float(preds[i].item())
                    if self._auc_transform is not None:
                        s = _inverse_auc_transform(raw_score, self._auc_transform)
                        if not np.isfinite(s):
                            s = 0.5
                        auc_min = float(self._auc_transform.get("min", 0.5))
                        auc_max = float(self._auc_transform.get("max", 1.0))
                        s = float(min(auc_max, max(auc_min, s)))
                    else:
                        s = raw_score
                        if not np.isfinite(s):
                            s = 0.5
                        if 0.0 <= s <= 1.0:
                            s = float(min(1.0, max(0.5, s)))
                    scored.append(s)

        # 3) Rehydrate in the original input order.
        for local_i, orig_idx in enumerate(valid_positions):
            results[orig_idx] = scored[local_i]
        return results
