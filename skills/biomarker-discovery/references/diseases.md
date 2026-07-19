# Disease catalogue — 13 phenotypes covered by the GAT models

This table is the source of truth for this release. It is a flat lookup the skill uses to resolve `<icd>` → folder, descriptive name, and external-cohort parquet path.

| disease_id (ICD-9 root) | Disease name | GAT folder on disk | MIMIC parquet | EHRShot parquet | NHANES labels? |
|---|---|---|---|---|---|
| 242  | Hyperthyroidism      | `GAT/242/`  | `Data/MIMIC/preprocessed/242_Hyperthyroidism_mimic.parquet`     | `Data/EHRShot/preprocessed/242_Hyperthyroidism_ehrshot.parquet`     | **yes** (any-thyroid proxy) |
| 2452 | Hashimoto            | `GAT/2542/` ⚠️ | `Data/MIMIC/preprocessed/2452_Hashimoto_mimic.parquet`            | `Data/EHRShot/preprocessed/2452_Hashimoto_ehrshot.parquet`            | **yes** (any-thyroid proxy; same label as 242) |
| 250  | T1D                  | `GAT/250/`  | `Data/MIMIC/preprocessed/250_T1D_mimic.parquet`                   | `Data/EHRShot/preprocessed/250_T1D_ehrshot.parquet`                   | **yes** |
| 277  | FMF                  | `GAT/277/`  | `Data/MIMIC/preprocessed/277_FMF_mimic.parquet`                   | `Data/EHRShot/preprocessed/277_FMF_ehrshot.parquet`                   | no |
| 340  | MS                   | `GAT/340/`  | `Data/MIMIC/preprocessed/340_MS_mimic.parquet`                    | `Data/EHRShot/preprocessed/340_MS_ehrshot.parquet`                    | no |
| 555  | Crohns               | `GAT/555/`  | `Data/MIMIC/preprocessed/555_Crohns_mimic.parquet`                | `Data/EHRShot/preprocessed/555_Crohns_ehrshot.parquet`                | no |
| 556  | UlcerativeColitis    | `GAT/556/`  | `Data/MIMIC/preprocessed/556_UlcerativeColitis_mimic.parquet`     | `Data/EHRShot/preprocessed/556_UlcerativeColitis_ehrshot.parquet`     | no |
| 5790 | Celiac               | `GAT/5790/` | `Data/MIMIC/preprocessed/5790_Celiac_mimic.parquet`               | `Data/EHRShot/preprocessed/5790_Celiac_ehrshot.parquet`               | no |
| 696  | Psoriasis            | `GAT/696/`  | `Data/MIMIC/preprocessed/696_Psoriasis_mimic.parquet`             | `Data/EHRShot/preprocessed/696_Psoriasis_ehrshot.parquet`             | **yes** |
| 7100 | SLE                  | `GAT/7100/` | `Data/MIMIC/preprocessed/7100_SLE_mimic.parquet`                  | `Data/EHRShot/preprocessed/7100_SLE_ehrshot.parquet`                  | no |
| 7101 | SystemicSclerosis    | `GAT/7101/` | `Data/MIMIC/preprocessed/7101_SystemicSclerosis_mimic.parquet`    | `Data/EHRShot/preprocessed/7101_SystemicSclerosis_ehrshot.parquet`    | no |
| 7102 | Sjogrens             | `GAT/7102/` | `Data/MIMIC/preprocessed/7102_Sjogrens_mimic.parquet`             | `Data/EHRShot/preprocessed/7102_Sjogrens_ehrshot.parquet`             | no |
| 714  | RheumatoidArthritis  | `GAT/714/`  | `Data/MIMIC/preprocessed/714_RheumatoidArthritis_mimic.parquet`   | `Data/EHRShot/preprocessed/714_RheumatoidArthritis_ehrshot.parquet`   | **yes** |

## Quick conventions

- **disease_id** is the ICD-9 root the rest of the codebase uses. Pass it as the `disease_id` argument to every `mcp__gat-multi-scorer__*` tool.
- **GAT folder typo** (⚠️): Hashimoto's checkpoint lives at `GAT/2542/` on disk, not `GAT/2452/`. The multi-MCP server's registry handles this transparently — call with `disease_id="2452"` and it routes to the `2542` checkpoint. Do NOT call with `"2542"`; that's the on-disk name, not the public ID.
- **target column in parquet**: always `icd_<folder>` (e.g., `icd_714`).
- **NHANES sparse coverage**: 5 folders have NHANES labels — `250` (T1D), `696` (Psoriasis), `714` (RheumatoidArthritis), `242` (Hyperthyroidism, any-thyroid proxy), and `2452` (Hashimoto, also any-thyroid proxy — shares the same label as 242 since NHANES can't distinguish them). Source of truth: `Code/preprocess_nhanes_external.py:NHANES_CASE_DEFINITIONS`. The skill's `--cohort nhanes` flag must reject any disease_id outside this 5-folder set.
- **Lab feature naming**: every column starts with `lab_` followed by a Clalit panel code and an aggregation suffix (typically `_last`; sometimes `_mean`/`_max`). For RA / MIMIC the 15 features are all `_last`. Get the *exact* per-disease list from `mcp__gat-multi-scorer__get_feature_list(disease_id)`.

## Default for the skill

When the user invokes `/biomarker-discovery` with no args, default to `disease_id="714"` (RheumatoidArthritis) and `cohort="mimic"`.
