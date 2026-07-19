# Expression DSL — what the GAT will accept

The GAT scores **binary expression trees** built from a fixed allowlist of **CBC (complete blood count) lab features** and four arithmetic operators (`+`, `-`, `*`, `/`). The biomarker grammar is CBC arithmetic — the same DSL used by the canonical orchestrator [`Code/e2e_2M_CBC_OPT_crohn.py`](../../../Code/e2e_2M_CBC_OPT_crohn.py). Anything outside this DSL fails to parse and the GAT returns `score=null`.

## Canonical CBC panel (14 features)

Bare medical names ↔ GAT-DSL form (with `lab_` prefix and `_last` aggregation suffix):

| Bare name (medical) | GAT-DSL form | Compartment |
|---|---|---|
| `BASO_pct`   | `lab_BASO_pct_last`   | WBC differential — basophils % |
| `EOS_pct`    | `lab_EOS_pct_last`    | WBC differential — eosinophils % |
| `LYMpct`     | `lab_LYMpct_last`     | WBC differential — lymphocytes % |
| `MONOpct`    | `lab_MONOpct_last`    | WBC differential — monocytes % |
| `NEUTpct`    | `lab_NEUTpct_last`    | WBC differential — neutrophils % |
| `WBC`        | `lab_WBC_last`        | Total leukocyte count |
| `HB`         | `lab_HB_last`         | RBC indices — hemoglobin |
| `HCT`        | `lab_HCT_last`        | RBC indices — hematocrit |
| `MCH`        | `lab_MCH_last`        | RBC indices — mean corpuscular Hb |
| `MCHC`       | `lab_MCHC_last`       | RBC indices — MCH concentration |
| `MCV`        | `lab_MCV_last`        | RBC indices — mean corpuscular volume |
| `RBC`        | `lab_RBC_last`        | RBC indices — total red cell count |
| `RDW`        | `lab_RDW_last`        | RBC indices — red cell distribution width |
| `PLT`        | `lab_PLT_last`        | Platelet count |

This 14-feature panel is the **canonical CBC reference** for the skill — and it's also the *exact* allowlist for **11 of the 13 disease GATs** (242, 250, 277, 340, 555, 556, 5790, 696, 7100, 7101, 7102). The only two exceptions are RA (714) and Hashimoto (2452), which add a 15th feature `lab_NEUT_abs_last` (absolute neutrophil count) on top of the 14-panel. The actual per-disease allowlist is always whatever `mcp__gat-multi-scorer__get_feature_list(disease_id)` returns at runtime, so the iteration loop never has to know these exceptions ahead of time.

## Hard rules (enforced by `Code/expr_graph_utils.py:string_to_data_obj`)

1. **Operators**: only `+`, `-`, `*`, `/`. Standard precedence (* and / before + and -). Parentheses to override.
2. **Leaves**: must be feature names from the per-disease allowlist (call `mcp__gat-multi-scorer__get_feature_list(disease_id)` once and cache it). Case-sensitive, exact match.
3. **No numeric literals**: `lab_PLT_last + 0.5` and `2 * lab_HB_last` both fail. Feature arithmetic only — even integer powers must be expressed as repeated multiplication (`lab_X * lab_X` instead of `lab_X**2`, but `**` itself is not supported either).
4. **No functions**: `log(...)`, `sqrt(...)`, `abs(...)`, `min/max(...)` — none are supported. Only `+ - * /`.
5. **No comparisons / conditionals**: no `==`, `>`, `?:`, `if`. Pure arithmetic.
6. **Implicit multiplication is NOT tolerated** at this layer. `lab_PLT_lastlab_NEUTpct_last` (LaTeX-style juxtaposition) parses as one unknown token and fails. Always use explicit `*`.
7. **Tree depth ≤ 3** is a soft prompt rule (not enforced at parse time, but deeper trees were not seen during GAT training and generalise poorly). "Depth" = number of operator nodes from root to deepest leaf.

## Five worked examples (all valid, all in the RA / MIMIC feature set)

```
1.  lab_NEUT_abs_last / lab_LYMpct_last
2.  (lab_PLT_last + lab_NEUT_abs_last) / lab_LYMpct_last
3.  (lab_RDW_last * lab_PLT_last) / (lab_LYMpct_last * lab_HB_last)
4.  lab_WBC_last - lab_NEUT_abs_last
5.  (lab_MCV_last + lab_RDW_last) / lab_HB_last
```

Each:
- Uses only `+ - * /`.
- All leaves are real columns in `714_RheumatoidArthritis_mimic.parquet`.
- Depth ≤ 3 (Example 3 is exactly depth 3: outer `/`, two children are `*` nodes).
- Parens are explicit where precedence matters.

## Common LLM mistakes — recognise and reject before sending

| Mistake | Why it fails | Fix |
|---|---|---|
| `lab_PLT_lastlab_NEUTpct_last` (LaTeX implicit mult) | Parser treats it as one unknown token. | Insert `*`: `lab_PLT_last * lab_NEUTpct_last`. |
| `lab_PLT_last + 0.5` | Numeric literal rejected. | Drop the constant. If you really need a baseline shift, you can't — express as a ratio instead. |
| `log(lab_HB_last)` | No functions. | Use the raw feature; the GAT was trained on raw values. |
| `lab_PLT_last**2` | `**` not in operator set. | `lab_PLT_last * lab_PLT_last` (depth still ≤ 3). |
| `lab_PLT_LAST` (wrong case) | Allowlist is case-sensitive. | Match the case from `get_feature_list(disease_id)` exactly. |
| `lab_NEUT_PCT_last` (typo) | Unknown token. | Use the exact suffix from the allowlist (likely `lab_NEUTpct_last`). |
| `((lab_A) / (lab_B)) / (lab_C / lab_D))` | Mismatched parens. | Recount; fix. |
| `lab_A / 0` | Numeric literal rejected (and would NaN at eval anyway). | Use a feature ratio. |

## Validation pipeline (what the skill runs before scoring)

```bash
python -c "
import json, sys
from gat_agent_tool.validation import parse_candidates
feats = json.loads(sys.argv[1])
ops   = ['+','-','*','/']
print('\n'.join(parse_candidates(sys.stdin.read(), feats, ops)))
" "$FEATURES_JSON" < raw_proposals.txt > validated.txt
```

`parse_candidates` strips Markdown bullets, code fences, and trailing comments; rejects expressions with numeric literals or unknown features; deduplicates while preserving order. After this filter, every survivor is *syntactically eligible* — but `string_to_data_obj` may still return None for malformed parens or arity issues, in which case the GAT score will be `null` and the skill should log `is_valid=false`.

## Score interpretation

- Range: `[0.5, 1.0]`.
- Higher is better.
- The GAT predicts a univariate-AUC proxy. It is NOT the real univariate AUC on any cohort — it's the model's learned estimate of how well the expression separates positives from negatives on the training distribution.
- The skill computes the *real* univariate AUC silently (via `scripts/silent_real_auc.py`) for logging only; the LLM must NEVER see real AUCs in the prompt (experimental control).
