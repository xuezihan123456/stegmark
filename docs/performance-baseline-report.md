# Performance Baseline Report

Generated on 2026-04-12 from the current `v0-1` worktree.

Method:

- Python: `3.13.5`
- Platform: `Windows-11-10.0.22631-SP0`
- Test image: `512x512 RGB`
- Native payload length: `2048` bits
- Codec payload size: `16384` bytes
- Each measurement: `8` rounds, report `mean / median / min`
- Legacy numbers come from local reference implementations that reproduce the pre-optimization logic in-process, so the comparison is reproducible from the current repo

## Results

### `bytes_to_bits`

| Implementation | Mean (ms) | Median (ms) | Min (ms) |
|---|---:|---:|---:|
| Legacy Python loop | 13.629 | 13.591 | 12.216 |
| Current `np.unpackbits` | 0.705 | 0.667 | 0.633 |

Speedup: `19.32x`

### native encode on `512x512`

| Implementation | Mean (ms) | Median (ms) | Min (ms) |
|---|---:|---:|---:|
| Legacy per-block Python path | 25.942 | 25.767 | 24.945 |
| Current batched block path | 15.378 | 14.928 | 13.933 |

Speedup: `1.69x`

## Notes

- The DCT path is measurably faster in this local CPU-only setup, but not yet near the earlier speculative `20x-50x` estimate from the analysis report.
- The codec vectorization win is large and stable.
- Raw machine-readable output is stored in [performance-baseline-report.json](D:/桌面/claude/项目/隐形水印工具/.worktrees/v0-1/docs/performance-baseline-report.json).
