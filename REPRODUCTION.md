# Code and data supplement

Readout adaptation in simulated arm reaching: learned compensation and performance after updates stop.

This package contains the six adaptive variants reported in the manuscript: dense-backbone, NCP-wired and FC-wired CfC with LMS, ESN-LMS, RBF-LMS, and dense-CfC RLS. Fixed-readout references and PD-only results are also supplied.

## Data and models

All three CfC configurations use final 400-epoch checkpoints. Learning-rate selection uses seeds 100–102; evaluation uses separate seeds 110–119. There are 65 checkpoint files: 39 CfC, 13 ESN and 13 RBF.

The rerun also evaluated a direction-indexed torque-offset condition, which was omitted from this report after the outcomes were inspected.

The supplied dataset contains 783 online protocols (135 selection, 333 evaluation, 315 new sensitivity) and 60 diagnostic reruns, totaling 843 protocols. The evaluation table includes 180 adaptive outcomes, 150 fixed-readout references and three PD-only runs. Sensitivity includes 405 adaptive outcomes, with 135 PD-gain-one selection outcomes reused; 349 are finite. The fixed field-on tests contain 120 cycles / 960 reaches, with additional null-field cycles for reproduction checks.

The manifests describe this dataset. All source hashes and checkpoint hashes are verified before execution. Original numerical modules include general-purpose helpers; use the portable entry point below for this six-method experiment.

## Environment

Use Python 3.11 and the supplied dependency lock. Keep the environment outside the extracted package and NAS storage:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/cfc-six-method-env
uv sync --locked --python 3.11 --project code --group dev
```

The locked torch wheel supports CUDA, but reproduction uses CPU and one torch thread per worker. The ncps distribution is 1.0.1.

## Reproduction

Run from this package root; each output directory must be new. `online --resume` can continue an interrupted run.

```bash
uv run --project code python scripts/reproduce_rerun400_package.py --stage verify --out checks
uv run --project code python scripts/reproduce_rerun400_package.py --stage figures --out figures_rerun
uv run --project code python scripts/reproduce_rerun400_package.py --stage online --workers 4 --out online_rerun
uv run --project code python scripts/reproduce_rerun400_package.py --stage diagnostics --workers 4 --out diagnostics_rerun
uv run --project code python scripts/reproduce_rerun400_package.py --stage snapshots --out snapshots_rerun
uv run --project code python scripts/reproduce_rerun400_package.py --stage offline --seeds 110 --out offline_check
```

`online` reruns all 783 protocols, checks the selected learning rates and compares every saved A/B/catch/washout curve. Persisted JSON represents non-finite numbers as null, so comparison preserves finite values, missing locations and failure flags, not the distinction between NaN and signed infinity. `diagnostics` regenerates all 60 learned snapshots and fixed tests and checks the supplied arrays. `snapshots` checks the nominal and fixed tests from the saved readouts. The 180 redundant Z feature arrays are omitted from NPZ containers; DERIVED_FEATURE_HASHES.json records their exact array-byte hashes for comparison with recomputed features.

`figures` regenerates four numerical figures and four tables using manifests/dataset_scope.json; its input hashes refer to the supplied CSV/JSON files. The three native TikZ diagrams are in paper_assets/. `offline` retrains the three CfC configurations for 400 epochs and refits ESN/RBF at recorded family settings; --seeds restricts it to a small check, and omitting --seeds selects all 65 models. The offline family-setting candidate grid is recorded but not rerun by this stage.

Fresh package validation and its actual scope are recorded in REPRODUCTION_VALIDATION.json and validation/. Availability of a command does not imply that it has been executed. Code is licensed under code/LICENSE.

Author: Jun Kobayashi. Preprint license: CC BY 4.0; code license: MIT.

Validation completed on this six-method package: all 783 online protocols and 60 diagnostic conditions were recomputed and matched; four figures and four tables reproduced; 30 unit tests passed. Seed 110 offline retraining/refitting matched all tensors in five models, and six snapshot conditions were checked. All 65 checkpoint hashes and 25 numerical-source hashes were verified. Stage reports record the file count present during execution; the validation reports were added afterward. PACKAGE_SHA256SUMS lists the current files.

## Distribution

This is the public code-and-data archive prepared for Zenodo. The source code will also be released at https://github.com/jkoba0512/cfc-online-adaptation. This archive is not an attachment to the bioRxiv revision. The archive release DOI must be obtained and cited before submission. No DOI has been issued by this preparation script.

Packaging update (2026-11-09): the signed-deviation schematic was added and Table 4 now uses its natural small font size. The previously recorded numerical reproduction checks remain applicable to unchanged numerical sources and data; this packaging update does not claim that full experiments were rerun.

## Packaging and portability

The public code-and-data archive and the anonymous journal supplement contain the same scientific arrays and reproduction code. Attribution and distribution descriptions differ. Local paths, Git identifiers, and author metadata are omitted from anonymous diagnostic records; no identified public-release link is included in the anonymous package.

Metadata uses relative paths. Relocating files can change metadata digests without changing scientific arrays or checkpoint tensors. The supplied source and checkpoint hashes remain available for verification.

## Direction-level fixed-readout analysis

`results/diagnostics/fixed_direction_pairs.csv` contains 480 paired direction-level errors (six methods, ten seeds, eight directions). `fixed_direction_summary.csv` reports the median of the ten within-seed ratios for each method and direction, rather than a ratio of separately computed medians. `fixed_direction_provenance.json` records the source hashes and metric definition. These are post-hoc summaries of the existing full-precision stored results, not new online experiments. Direction index 7 is the last in the fixed training order.

Regenerate these summaries from the package root:

```bash
python scripts/summarize_fixed_direction_errors.py --package-root . --out direction_summary_rerun
```
