# Environment

This project uses `uv`.

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

If you prefer to keep the virtual environment outside the repository, set
`UV_PROJECT_ENVIRONMENT` before running `uv sync` or `uv run`.
