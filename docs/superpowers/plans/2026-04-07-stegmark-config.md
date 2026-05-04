# StegMark Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent config file support and a `stegmark config` CLI with `show`, `set`, and `reset`.

**Architecture:** Keep config handling isolated in `src/stegmark/config.py`, with explicit precedence: CLI args > environment variables > config file > defaults. The CLI should expose config management without coupling it to engine internals.

**Tech Stack:** Python 3.10+, Click, stdlib `tomllib`

---

### Task 1: Add config file parsing and persistence

**Files:**
- Modify: `src/stegmark/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from stegmark.config import load_config, reset_config_file, save_config_value


def test_save_and_load_config_value(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    save_config_value(config_path, "engine", "native")
    config = load_config(config_path=config_path)
    assert config.engine == "native"


def test_reset_config_file_removes_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    save_config_value(config_path, "engine", "native")
    reset_config_file(config_path)
    assert not config_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL because config persistence helpers do not exist yet

- [ ] **Step 3: Implement config read/write helpers**

```python
def default_config_path() -> Path:
    ...


def load_config(*, config_path: Path | None = None) -> StegMarkConfig:
    ...


def save_config_value(config_path: Path, key: str, value: str) -> None:
    ...


def reset_config_file(config_path: Path) -> None:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stegmark/config.py tests/test_config.py
git commit -m "feat: add config persistence"
```

### Task 2: Add `stegmark config` CLI commands

**Files:**
- Modify: `src/stegmark/cli.py`
- Create: `tests/test_config_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing CLI tests**

```python
from click.testing import CliRunner

from stegmark.cli import main


def test_config_show_outputs_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STEGMARK_CONFIG", str(tmp_path / "config.toml"))
    runner = CliRunner()
    runner.invoke(main, ["config", "set", "engine", "native"])
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0
    assert '"engine": "native"' in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_cli.py -q`
Expected: FAIL because the config command does not exist yet

- [ ] **Step 3: Implement `config show`, `config set`, and `config reset`**

```python
@main.group()
def config() -> None:
    ...
```

- [ ] **Step 4: Update README usage**

```markdown
stegmark config show
stegmark config set engine native
stegmark config reset
```

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_config_cli.py tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/stegmark/cli.py tests/test_config_cli.py README.md
git commit -m "feat: add config cli"
```
