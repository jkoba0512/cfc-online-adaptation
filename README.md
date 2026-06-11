# cfc-online-adaptation

Climbing-fiber-like online adaptation of a CfC readout for motor-control
recalibration.

This repository contains simulation code for a two-link planar reaching task with
a frozen CfC core and online adaptation of a linear readout. Manuscript sources,
paper figures, and generated experiment outputs are not tracked in this public
code repository.

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
