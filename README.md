# cfc-online-adaptation

[![CI](https://github.com/jkoba0512/cfc-online-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/jkoba0512/cfc-online-adaptation/actions/workflows/ci.yml)

Climbing-fiber-like online adaptation of a CfC readout for motor-control
recalibration.

This repository contains simulation code for a two-link planar reaching task with
a frozen CfC core and online adaptation of a linear readout. Manuscript sources,
paper figures, and generated experiment outputs are not tracked in this public
code repository.

## What is implemented

- A two-link planar arm model with velocity-dependent curl force fields.
- Minimum-jerk center-out reaching tasks.
- A CfC inverse model with a frozen recurrent core and an adaptable linear
  readout.
- Online readout adaptation with LMS and RLS-style update rules.
- Baseline controllers, after-effect analysis, perturbation sweeps, and tests.

## Repository layout

- `code/`: Python package, experiment scripts, and tests.
- `docs/`: Public project documentation.

## Quick start

```bash
cd code
uv sync
uv run pytest -q
uv run ruff check .
```

The main package implements the arm dynamics, task generation, CfC inverse model,
linear LMS/RLS readouts, closed-loop adaptation protocols, and analysis helpers.

## Experiment scripts

Run scripts from `code/`:

```bash
cd code
uv run python scripts/run_step2_b0_smoke.py
uv run python scripts/run_step3_adaptation_smoke.py
uv run python scripts/run_step4_baselines_aftereffect.py
uv run python scripts/run_step5_phase1_final.py
uv run python scripts/run_phase2_step1_wiredcfc_smoke.py
uv run python scripts/run_phase2_step2_wiredcfc_formal_2a.py
uv run python scripts/run_phase2_step3_perturbation_sweep.py
uv run python scripts/run_phase2_step4_rls_washout_mechanism.py
uv run python scripts/run_phase2_step6_readout_boundary.py
```

Scripts write generated artifacts under `outputs/`, which is intentionally
ignored by Git.

## Tests and linting

```bash
cd code
uv run pytest -q
uv run ruff check .
```

## Tracked and untracked material

This repository tracks source code, tests, and public documentation. It does not
track generated experiment outputs, manuscript sources, paper figures, local
agent metadata, virtual environments, or LaTeX build artifacts.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
