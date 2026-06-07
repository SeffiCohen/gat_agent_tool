"""Independent ORACLE implementation of GatScorerRegistry.

This file mirrors the public contract of `gat_agent_tool.registry.GatScorerRegistry`
but is intentionally written in a different internal style so a verifier script
can diff the two and surface any disagreement (the same pattern used by
`Code/_external_orchestrator_oracle.py` against `run_external_methods_eval_all.py`).

Intentional divergences from the likely IMPLEMENTER style:
  * No walrus operator — explicit `cached = ...; if cached is not None: return cached`.
  * `os.listdir` + `os.path.join` + `os.path.isdir`/`os.path.isfile` instead of
    `pathlib.Path.iterdir()` / generator expressions.
  * Reverse-override built with an explicit `for` loop, not a dict comprehension.
  * `info()` builds its dict by hand from named ModelInfo attributes (so renaming
    a field in core.py would diverge between IMPLEMENTER's `dataclasses.asdict`
    and this longhand path).
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover — avoid pulling torch at import time
    from .core import GatScorerTool


logger = logging.getLogger(__name__)


class GatScorerRegistry:
    """Multi-disease registry that lazy-loads `GatScorerTool` instances.

    Public method signatures + behavior must match the IMPLEMENTER exactly. See
    the docstrings on each method for the contract.
    """

    def __init__(
        self,
        gat_root: Optional[Path] = None,
        registry_config: Optional[Dict[str, str]] = None,
        checkpoint_filename: str = "best_by_loss.pt",
        device: str = "auto",
        lazy: bool = True,
        disk_folder_override: Optional[Dict[str, str]] = None,
    ) -> None:
        # ---- argument cross-validation: exactly one of root / config -----
        root_set = gat_root is not None
        config_set = registry_config is not None
        if root_set and config_set:
            raise ValueError(
                "GatScorerRegistry: pass either gat_root OR registry_config, not both"
            )
        if (not root_set) and (not config_set):
            raise ValueError(
                "GatScorerRegistry: pass either gat_root OR registry_config (got neither)"
            )

        # Default override mirrors the on-disk typo for ICD 2452 (folder 2542/).
        if disk_folder_override is None:
            self._overrides: Dict[str, str] = {"2452": "2542"}
        else:
            self._overrides = dict(disk_folder_override)

        self._device: str = device
        self._checkpoint_filename: str = checkpoint_filename
        self._index: Dict[str, Path] = {}
        self._cache: Dict[str, "GatScorerTool"] = {}
        # One global lock guards _cache mutation. A per-id lock would also work,
        # but a single lock is simpler to reason about for the oracle.
        self._lock: threading.Lock = threading.Lock()

        if root_set:
            self._discover_from_root(gat_root)  # type: ignore[arg-type]
        else:
            # Explicit map: caller supplies advertised_id -> checkpoint path.
            for advertised_id, ckpt_path in registry_config.items():  # type: ignore[union-attr]
                self._index[str(advertised_id)] = Path(str(ckpt_path))

        # Eager construction is just "ask for every advertised id".
        if not lazy:
            for did in self.list_diseases():
                self.get(did)

    # ------------------------------------------------------------------
    # Discovery (auto-mode only)
    # ------------------------------------------------------------------
    def _discover_from_root(self, gat_root: Path) -> None:
        root_str = str(gat_root)
        if not os.path.isdir(root_str):
            raise FileNotFoundError(
                f"GatScorerRegistry: gat_root not found or not a directory: {root_str}"
            )

        # Build on-disk -> advertised mapping with an explicit loop (oracle
        # divergence: implementer would likely use a dict comprehension).
        on_disk_to_advertised: Dict[str, str] = {}
        for advertised, on_disk in self._overrides.items():
            on_disk_to_advertised[on_disk] = advertised

        # `os.listdir` + manual `os.path` checks — different code path than
        # `pathlib.Path.iterdir()`. Sort to lock in deterministic ordering.
        try:
            raw_entries = os.listdir(root_str)
        except OSError as exc:  # pragma: no cover — defense in depth
            raise FileNotFoundError(
                f"GatScorerRegistry: could not list gat_root {root_str}: {exc}"
            ) from exc

        for entry_name in sorted(raw_entries):
            if entry_name.startswith("."):
                continue  # skip dotfiles (.DS_Store, .git, ...)
            entry_path = os.path.join(root_str, entry_name)
            if not os.path.isdir(entry_path):
                continue
            ckpt_full = os.path.join(entry_path, "trained_models", self._checkpoint_filename)
            if not os.path.isfile(ckpt_full):
                continue

            # Apply reverse-override: on-disk folder may advertise as a
            # different disease_id (e.g. 2542/ -> "2452").
            advertised_id = on_disk_to_advertised.get(entry_name, entry_name)
            self._index[advertised_id] = Path(ckpt_full)

    # ------------------------------------------------------------------
    # Read-only introspection
    # ------------------------------------------------------------------
    def list_diseases(self) -> List[str]:
        # Fresh sorted list every call so the caller can mutate without harm.
        return sorted(self._index.keys())

    # ------------------------------------------------------------------
    # Cached construction
    # ------------------------------------------------------------------
    def get(self, disease_id: str) -> "GatScorerTool":
        if disease_id not in self._index:
            raise KeyError(
                f"unknown disease_id={disease_id!r}; available={sorted(self._index)}"
            )

        # Fast path: avoid lock acquisition when already cached. Spelled out
        # without walrus to make the oracle's control flow obvious.
        cached = self._cache.get(disease_id)
        if cached is not None:
            return cached

        with self._lock:
            # Recheck under the lock — another thread may have constructed
            # while we were blocked. Critical for thread-safety test.
            cached_again = self._cache.get(disease_id)
            if cached_again is not None:
                return cached_again

            # LAZY import — kept inside get() so module import doesn't pull
            # torch. This MUST mirror the IMPLEMENTER (constructor must not
            # import .core).
            from .core import GatScorerTool  # noqa: WPS433 — intentional lazy

            checkpoint_path = self._index[disease_id]
            logger.info(
                "GatScorerRegistry: constructing GatScorerTool for disease_id=%s "
                "(checkpoint=%s, device=%s)",
                disease_id, str(checkpoint_path), self._device,
            )
            # Pass the str() form so caller-side `os.path.exists()` and friends
            # work uniformly (Path objects are also accepted by GatScorerTool,
            # but the production code paths uses `str(...)` internally anyway).
            new_tool = GatScorerTool(str(checkpoint_path), device=self._device)
            self._cache[disease_id] = new_tool
            return new_tool

    # ------------------------------------------------------------------
    # Thin convenience delegators
    # ------------------------------------------------------------------
    def score(self, disease_id: str, expression: str) -> Optional[float]:
        scorer = self.get(disease_id)
        return scorer.score(expression)

    def score_batch(self, disease_id: str, expressions: List[str]) -> List[Optional[float]]:
        scorer = self.get(disease_id)
        return scorer.score_batch(expressions)

    def feature_names(self, disease_id: str) -> List[str]:
        scorer = self.get(disease_id)
        return scorer.feature_names

    def info(self, disease_id: str) -> Dict[str, Any]:
        scorer = self.get(disease_id)
        model_info = scorer.info()  # ModelInfo dataclass
        # Build the dict by hand from named attributes — different code path
        # than `dataclasses.asdict(model_info)`. If a field is renamed in
        # core.py the IMPLEMENTER's asdict() output will diverge from this.
        out: Dict[str, Any] = {
            "model_type": model_info.model_type,
            "num_features": model_info.num_features,
            "num_operators": model_info.num_operators,
            "max_depth": model_info.max_depth,
            "graph_format_version": model_info.graph_format_version,
            "has_auc_transform": model_info.has_auc_transform,
            "checkpoint_path": model_info.checkpoint_path,
            "disease_id": disease_id,
        }
        return out

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def unload(self, disease_id: str) -> None:
        with self._lock:
            # `pop` with default — silent no-op when absent.
            self._cache.pop(disease_id, None)
