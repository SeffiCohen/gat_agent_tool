# Model License — Trained GAT Checkpoints

## Scope

This license applies to the **13 trained Graph Attention Network (GAT) checkpoints**
distributed under the `GAT/` directory of this repository (one checkpoint folder per
disease ICD code). It governs the released model weights only. The accompanying
source code is licensed separately under Apache-2.0 (see `LICENSE`).

## License

The trained GAT checkpoints under `GAT/` are released under the
**Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

Canonical license text: <https://creativecommons.org/licenses/by-nc/4.0/>

Under CC BY-NC 4.0 you are free to **share** (copy and redistribute the weights in any
medium or format) and **adapt** (remix, transform, and build upon the weights), under
the following terms:

- **Attribution** — You must give appropriate credit (cite the companion paper, see
  below), provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may **not** use the weights for commercial purposes.

No additional restrictions: you may not apply legal terms or technological measures
that legally restrict others from doing anything the license permits.

## Provenance

These weights were **trained inside the Clalit Health Services data boundary**. Only
the **learned model weights** were exported from that environment. The released
checkpoints contain:

- **No patient rows** — no individual-level records of any kind.
- **No individual CBC (complete blood count) values** — no raw or derived per-patient
  laboratory measurements.
- **No expression-level AUC labels** — none of the per-expression performance labels
  used during training are included.

In other words, the public artifact is the distilled, parameter-only output of a model
trained on private electronic health record (EHR) data; the underlying patient data
never leaves the Clalit boundary and is not redistributed here.

## Permitted use

Use of these checkpoints is restricted to **non-commercial** purposes (e.g. academic
research, education, methods development, and reproducibility). Commercial use requires
separate written permission from the authors.

## Citation

If you use these trained GAT checkpoints, please cite the companion paper:

> **Distilling private EHR evidence into a public agentic tool for CBC biomarker discovery.**
