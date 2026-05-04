# Contributing

## Development Setup

Requirements:

- Python 3.10+

Install the project in editable mode:

```bash
python -m pip install -e .[dev]
```

Optional extras:

```bash
python -m pip install -e .[hidden]
python -m pip install -e .[train]
python -m pip install -e .[trustmark]
```

## Development Workflow

Run the local quality checks before committing:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src
python -m build
```

## Commit Style

Use Conventional Commits when possible:

- `feat: add benchmark gate automation`
- `fix: handle hidden runtime weight lookup`
- `docs: update benchmark usage`
- `test: cover benchmark comparison outputs`
- `chore: bump release metadata`

## Pull Requests

Keep pull requests focused. A good PR should:

- solve one coherent problem
- include tests for new behavior
- keep documentation in sync with code
- avoid unrelated refactors

## Benchmark and Model Changes

If you change benchmark thresholds, attack implementations, or hidden-model behavior:

- update benchmark tests
- explain the change in `CHANGELOG.md`
- include the exact verification commands you ran

