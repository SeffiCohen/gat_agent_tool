"""gat_agent_tool.registry — multi-disease wrapper around `GatScorerTool`.

Most consumers of this package score expressions against a single trained GAT
checkpoint and use :class:`gat_agent_tool.core.GatScorerTool` directly. The
registry exists for the *multi-disease* path: the BD_Paper repo trains one GAT
per ICD code (currently 13 phenotypes, one folder per code under
``GAT/<icd>/trained_models/``), and downstream agents — the MCP server, the
OpenAI / HF iterative loops, the orchestration scripts — need to switch between
diseases without paying the per-call torch-load cost.

:class:`GatScorerRegistry` solves that with:

  * **Auto-discovery** — point ``gat_root`` at the BD_Paper ``GAT/`` directory
    and the registry walks the tree, finding every
    ``<root>/<folder>/trained_models/<checkpoint_filename>`` file.
  * **Lazy loading** — by default no checkpoint is opened until the disease is
    first requested; pass ``lazy=False`` to eagerly construct every scorer at
    init time (useful for warm-starting a long-running server).
  * **Thread-safe caching** — once loaded, a `GatScorerTool` is reused for the
    lifetime of the registry. The cache is guarded by a single
    :class:`threading.Lock` with double-checked locking, so concurrent first
    requests for the same disease still construct the scorer exactly once.
  * **Disk-folder override** — some on-disk folder names disagree with the
    public ICD code we want to advertise. The canonical example is the
    Hashimoto checkpoint, which lives in ``GAT/2542/`` (typo) but should be
    served as ``disease_id="2452"``. The default override
    ``{"2452": "2542"}`` handles this transparently; pass ``{}`` to disable
    or supply your own ``{advertised: on_disk}`` mapping for new typos.

If you do not need multi-disease switching — i.e. you already know the single
checkpoint path and never want to score anything else — use
:class:`gat_agent_tool.core.GatScorerTool` directly and skip this module.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    from .core import GatScorerTool


logger = logging.getLogger(__name__)


# Default disk-folder override: {advertised_disease_id: on_disk_folder_name}.
# Mirrors `GAT_FOLDER_OVERRIDE` in Code/run_external_methods_eval_all.py — the
# ICD 2452 (Hashimoto) checkpoint sits in a `2542/` directory due to a digit
# swap when the folder was first created; users still want to look it up by the
# correct ICD code.
_DEFAULT_DISK_FOLDER_OVERRIDE: Dict[str, str] = {"2452": "2542"}


class GatScorerRegistry:
    """Multi-disease registry of `GatScorerTool` instances with lazy loading.

    Exactly one of ``gat_root`` (auto-discover from a directory tree) or
    ``registry_config`` (explicit ``{disease_id: checkpoint_path}`` mapping)
    must be provided; supplying both or neither raises ``ValueError``.

    Parameters
    ----------
    gat_root : pathlib.Path or str, optional
        Root directory holding one subfolder per disease, each containing
        ``trained_models/<checkpoint_filename>``. Skipped subfolders: hidden
        ones (leading ``.``) and any without a matching checkpoint file.
    registry_config : dict, optional
        Pre-built ``{disease_id: checkpoint_path}`` mapping. No filesystem
        validation is performed — the caller owns path correctness.
    checkpoint_filename : str, default ``"best_by_loss.pt"``
        Filename to look for under each ``<folder>/trained_models/`` directory
        when auto-discovering.
    device : str, default ``"auto"``
        Passed verbatim to every constructed :class:`GatScorerTool`.
    lazy : bool, default ``True``
        When ``False``, eagerly construct a scorer for every advertised
        disease in ``__init__``. Otherwise scorers are built on first request.
    disk_folder_override : dict, optional
        ``{advertised_disease_id: on_disk_folder_name}`` mapping. Default is
        ``{"2452": "2542"}`` (the Hashimoto typo); pass ``{}`` to disable.
    """

    def __init__(
        self,
        gat_root: Optional[Union[Path, str]] = None,
        registry_config: Optional[Dict[str, str]] = None,
        checkpoint_filename: str = "best_by_loss.pt",
        device: str = "auto",
        lazy: bool = True,
        disk_folder_override: Optional[Dict[str, str]] = None,
    ) -> None:
        if gat_root is not None and registry_config is not None:
            raise ValueError(
                "GatScorerRegistry: pass exactly one of gat_root or "
                "registry_config, not both."
            )
        if gat_root is None and registry_config is None:
            raise ValueError(
                "GatScorerRegistry: must pass either gat_root (for auto-"
                "discovery) or registry_config (explicit mapping)."
            )

        self._device = device
        self._checkpoint_filename = checkpoint_filename
        self._overrides: Dict[str, str] = (
            dict(_DEFAULT_DISK_FOLDER_OVERRIDE)
            if disk_folder_override is None
            else dict(disk_folder_override)
        )
        self._lock = threading.Lock()
        self._cache: Dict[str, "GatScorerTool"] = {}

        if gat_root is not None:
            self._index = self._discover(Path(gat_root))
        else:
            assert registry_config is not None  # for type-checkers
            self._index = {str(k): str(v) for k, v in registry_config.items()}

        if not lazy:
            for did in list(self._index.keys()):
                self.get(did)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _discover(self, gat_root: Path) -> Dict[str, str]:
        """Walk ``gat_root`` looking for ``<folder>/trained_models/<ckpt>``."""
        if not gat_root.is_dir():
            raise FileNotFoundError(
                f"GatScorerRegistry: gat_root does not exist or is not a "
                f"directory: {gat_root}"
            )
        # Reverse the override so we can map on-disk folder -> advertised id.
        on_disk_to_advertised: Dict[str, str] = {
            v: k for k, v in self._overrides.items()
        }
        index: Dict[str, str] = {}
        entries = sorted(
            p.name
            for p in gat_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        for entry in entries:
            ckpt = gat_root / entry / "trained_models" / self._checkpoint_filename
            if ckpt.is_file():
                advertised = on_disk_to_advertised.get(entry, entry)
                index[advertised] = str(ckpt)
            else:
                logger.debug(
                    "GatScorerRegistry: skipping %s (no checkpoint at %s)",
                    entry, ckpt,
                )
        return index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_diseases(self) -> List[str]:
        """Return advertised disease IDs in sorted order. Fresh list each call."""
        return sorted(self._index.keys())

    def get(self, disease_id: str) -> "GatScorerTool":
        """Return the cached `GatScorerTool` for ``disease_id``, building it on
        first request. Thread-safe via double-checked locking — concurrent
        first requests still construct the scorer exactly once. Raises
        ``KeyError`` (with the available IDs in the message) if unknown."""
        if disease_id not in self._index:
            raise KeyError(
                f"unknown disease_id={disease_id!r}; "
                f"available={sorted(self._index)}"
            )
        cached = self._cache.get(disease_id)
        if cached is not None:
            logger.debug("GatScorerRegistry: cache hit for %s", disease_id)
            return cached
        with self._lock:
            cached = self._cache.get(disease_id)
            if cached is not None:
                return cached
            logger.debug(
                "GatScorerRegistry: cache miss for %s, loading checkpoint",
                disease_id,
            )
            from .core import GatScorerTool  # lazy: keep torch out of __init__
            tool = GatScorerTool(self._index[disease_id], device=self._device)
            self._cache[disease_id] = tool
            return tool

    def score(self, disease_id: str, expression: str) -> Optional[float]:
        """Score one expression against ``disease_id``'s GAT. Returns the
        predicted AUC in ``[0.5, 1.0]`` or ``None`` if the expression cannot
        be parsed. Raises ``KeyError`` for unknown ``disease_id``."""
        return self.get(disease_id).score(expression)

    def score_batch(
        self, disease_id: str, expressions: List[str]
    ) -> List[Optional[float]]:
        """Score a list of expressions against ``disease_id``'s GAT. Output
        order matches input; un-parseable entries come back as ``None``.
        Raises ``KeyError`` for unknown ``disease_id``."""
        return self.get(disease_id).score_batch(expressions)

    def feature_names(self, disease_id: str) -> List[str]:
        """Return the lab feature names ``disease_id``'s GAT was trained on.
        Raises ``KeyError`` for unknown ``disease_id``."""
        return self.get(disease_id).feature_names

    def info(self, disease_id: str) -> Dict[str, Any]:
        """Return a dict of metadata for ``disease_id``'s loaded GAT, with the
        ``ModelInfo`` dataclass flattened to a plain dict (so the MCP layer
        can serialize it) and a ``disease_id`` key added. Raises ``KeyError``
        for unknown ``disease_id``."""
        meta = self.get(disease_id).info()
        # Convert dataclass-like ModelInfo into a plain dict without importing
        # `dataclasses`, since the FakeScorer in the test suite returns a
        # SimpleNamespace-style object instead of a real dataclass.
        out: Dict[str, Any] = {
            k: getattr(meta, k)
            for k in (
                "model_type",
                "num_features",
                "num_operators",
                "max_depth",
                "graph_format_version",
                "has_auc_transform",
                "checkpoint_path",
            )
        }
        out["disease_id"] = disease_id
        return out

    def unload(self, disease_id: str) -> None:
        """Drop ``disease_id`` from the cache so the next ``get()`` rebuilds
        it. No-op if the disease was never loaded; never raises."""
        with self._lock:
            self._cache.pop(disease_id, None)


__all__ = ["GatScorerRegistry"]
