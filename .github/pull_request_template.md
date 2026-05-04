## Summary

<!-- What problem does this PR solve? -->

## Related Issues

<!-- Link issues, discussions, or benchmark reports. -->

## Changes

<!-- Keep this focused on the substantive changes. -->

## Engine and Runtime Impact

- Engine(s): `native` / `hidden` / `trustmark` / `auto` / `benchmark only` / `none`
- Platform(s) checked: <!-- Windows / macOS / Linux -->
- Runtime details: <!-- Python version, CPU/GPU, optional extras like hidden/trustmark/gpu -->

## Verification

- Tests run:
  - [ ] `python -m pytest -q`
  - [ ] `python -m ruff check .`
  - [ ] `python -m mypy src`
  - [ ] `python -m build`
  - [ ] Other: <!-- list additional commands or explain skips -->

## Benchmark Context

<!-- Required when touching benchmark logic, attack implementations, thresholds, or engine robustness. -->

- Relevant: [ ] yes [ ] no
- Command(s):
- Before/after metrics: <!-- avg bit accuracy, PSNR, worst attack, etc. -->
- Report artifact or summary:

## Checklist

- [ ] Tests cover the new behavior, or I explained why not
- [ ] Docs and examples stay in sync
- [ ] `CHANGELOG.md` updated if benchmark/model behavior changed
- [ ] No unrelated refactors are mixed in