# Contributing

Use uv for environments and Ruff for formatting and linting.

```sh
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
```

Keep comments short and use sentence case without punctuation. Prefer clear names and small functions over explanatory comments.

## Stacked changes

Keep each pull request independently reviewable. The intended stack is:

1. `feature/docs-tooling` adds project metadata, uv, Ruff, and documentation
2. `feature/experiment-scripts` adds reusable detection code and scripts
3. `feature/tests` adds the test suite and validation wiring

Open each branch against the previous branch in the stack. Merge from the bottom upward after checks pass.
