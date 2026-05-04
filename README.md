# StegMark

**StegMark** — 轻量级不可见水印工具包，支持频域嵌入、可逆水印、抗翻拍、AIGC 溯源、零知识证明等高级功能。

**StegMark** — A lightweight invisible watermark toolkit featuring frequency-domain embedding, reversible watermarks, screen-capture resilience, AIGC provenance, and zero-knowledge proof verification.

当前开发版本 / Current development version: `0.3.0a1`

---

## 功能概览 / Features

### 核心水印引擎 / Core Watermark Engines

| 引擎 / Engine | 状态 / Status | 安装 / Install | 说明 / Description |
|---|---|---|---|
| `native` (DCT) | ✅ 就绪 / Ready | `pip install stegmark` | 基于 DCT 频域的离线嵌入，仅需 NumPy + Pillow / DCT-based offline embedding, requires only NumPy + Pillow |
| `hidden` (HiDDeN ONNX) | ⚠️ 需模型权重 / Needs weights | `pip install stegmark[hidden]` | 深度学习风格的自托管水印 / Deep-learning style self-hosted watermark |
| `reversible` (LSB) | ✅ 就绪 / Ready | `pip install stegmark` | 基于 LSB 的可完全恢复水印 / LSB-based fully reversible watermark |
| `screen_resilient` (FFT) | ✅ 就绪 / Ready | `pip install stegmark` | 傅里叶-梅林变换域嵌入，抗旋转缩放 / Log-polar FFT embedding, rotation & scale resilient |
| `trustmark` | ✅ 就绪 / Ready | `pip install stegmark[trustmark]` | Adobe TrustMark 适配器 / Adobe TrustMark adapter |

### 高级功能模块 / Advanced Modules

| 模块 / Module | 文件 / File | 说明 / Description |
|---|---|---|
| **溯源链** / Provenance Chain | `core/provenance.py` | 多层 DCT 嵌入，支持 4 层溯源记录 / Multi-layer DCT embedding, up to 4 provenance layers |
| **零知识证明** / ZK Proof | `core/zk_proof.py` | 基于 SHA-256 承诺的水印所有权验证 / SHA-256 commitment-based watermark ownership verification |
| **感知强度** / Perceptual Strength | `core/perceptual_strength.py` | JND 自适应嵌入强度，高纹理区增强、平坦区减弱 / JND-adaptive embedding strength: boost in textured regions, reduce in flat areas |
| **AIGC 指纹** / AIGC Fingerprint | `core/aigc_fingerprint.py` | AI 生成内容溯源元数据嵌入，兼容 C2PA 格式 / AIGC provenance metadata embedding, C2PA compatible |
| **对抗训练** / Adversarial Training | `training/adversarial.py` | PGD 对抗攻击训练，提升水印鲁棒性 / PGD adversarial training for watermark robustness |
| **取证分析** / Forensics | `evaluation/forensics.py` | 差值热力图、DCT 修改图、频域分析、鲁棒性热力图 / Diff heatmap, DCT modification map, frequency analysis, robustness heatmap |
| **攻击模拟** / Attack Simulation | `evaluation/attacks.py` | JPEG 压缩、缩放、裁剪、模糊、噪声、亮度、丢弃等攻击 / JPEG, resize, crop, blur, noise, brightness, dropout attacks |
| **质量度量** / Quality Metrics | `evaluation/metrics.py` | PSNR、比特准确率、差异图 / PSNR, bit accuracy, diff image |

---

## 安装 / Installation

### 基础安装 / Core

```bash
pip install .
```

### 开发环境 / Development

```bash
pip install -e .[dev]
```

### 可选依赖 / Optional Dependencies

```bash
# TrustMark 适配器 / TrustMark adapter
pip install -e .[trustmark]

# 训练栈（PyTorch + ONNX）/ Training stack (PyTorch + ONNX)
pip install -e .[train]

# Hidden 运行时（ONNX Runtime）/ Hidden runtime (ONNX Runtime)
pip install -e .[hidden]

# GPU 加速 / GPU acceleration
pip install -e .[gpu]
```

> **注意 / Note**: 在 Python 3.13 上，`train` 和 `trustmark` 的 NumPy 约束可能冲突，建议使用独立环境。
> On Python 3.13, the `train` and `trustmark` extras may have conflicting NumPy constraints — use separate environments.

---

## CLI 快速上手 / CLI Quick Start

### 嵌入水印 / Embed a Watermark

```bash
# 基本嵌入 / Basic embed
stegmark embed input.png -m "Alice 2026" -o output.png

# 指定格式和质量 / Specify format and quality
stegmark embed input.png -m "Alice 2026" --format jpeg --quality 90 --overwrite -o output.jpg

# 生成对比报告 / Generate compare report
stegmark embed input.png -m "Alice 2026" --compare -o output.png

# 嵌入原始十六进制载荷 / Embed raw hex payload
stegmark embed input.png --bits deadbeef -o output.png

# 批量嵌入 / Batch embed
stegmark embed ./photos -m "Alice 2026" -o ./protected -r -w 4

# 指定引擎 / Specify engine
stegmark embed input.png -m "Alice 2026" -e native
stegmark embed input.png -m "Alice 2026" -e reversible
stegmark embed input.png -m "Alice 2026" -e screen_resilient
```

### 提取水印 / Extract a Watermark

```bash
# 提取消息 / Extract message
stegmark extract output.png

# 提取原始载荷 / Extract raw payload
stegmark extract output.png --mode bits

# 指定引擎提取 / Extract with specific engine
stegmark extract output.png -e hidden
```

### 验证水印 / Verify a Watermark

```bash
stegmark verify output.png -m "Alice 2026"
```

### 查看水印信息 / Inspect Watermark Info

```bash
stegmark info output.png
```

### 基准测试 / Benchmark

```bash
# 基本基准测试 / Basic benchmark
stegmark benchmark input.png -m "Alice 2026" --attacks jpeg_q90,brightness_1.3 --output-dir ./report --report-format html

# 多引擎对比 / Multi-engine comparison
stegmark benchmark input.png -m "Alice 2026" --engines native,hidden --attacks jpeg_q90 --json

# 设置质量门限 / Set quality gates
stegmark benchmark input.png -m "Alice 2026" --attacks jpeg_q90 --min-average-bit-accuracy 0.95
```

### 配置管理 / Config Management

```bash
stegmark config show
stegmark config set engine native
stegmark config reset --yes
```

---

## Python API / Python API

### 基础用法 / Basic Usage

```python
import stegmark

# 嵌入 / Embed
result = stegmark.embed("input.png", "Alice 2026", output="output.png", engine="native")
print(result.output_path)

# 提取 / Extract
extracted = stegmark.extract("output.png", engine="native")
print(extracted.message)

# 验证 / Verify
verified = stegmark.verify("output.png", "Alice 2026", engine="native")
print(verified.matched)

# 基准测试 / Benchmark
benchmark = stegmark.benchmark(
    "input.png", "Alice 2026",
    engine="native",
    attacks=["jpeg_q90", "brightness_1.3"],
)
print(benchmark.attack_results["jpeg_q90"].bit_accuracy)

# 原始载荷模式 / Raw payload mode
raw = stegmark.embed("input.png", bits="deadbeef", output="bits-output.png", engine="native")
decoded = stegmark.extract("bits-output.png", engine="native")
print(decoded.payload_hex)
```

### 会话模式 / Reusable Session

```python
from stegmark import StegMark

with StegMark(engine="native", strength=1.0) as client:
    client.embed("input.png", "Studio X", output="session.png")
    extracted = client.extract("session.png")
    print(extracted.message)
```

### 可逆水印 / Reversible Watermark

```python
from stegmark.core.reversible import ReversibleEngine

engine = ReversibleEngine()

# 嵌入并支持完全恢复 / Embed with full reversibility
watermarked = engine.encode(image, "sensitive data")

# 提取 / Extract
result = engine.decode(watermarked)
print(result.message)

# 恢复原始图像 / Restore original image
restored = engine.restore(watermarked)
```

### 溯源链 / Provenance Chain

```python
from stegmark.core.provenance import multi_layer_embed, multi_layer_extract, build_provenance_entry

# 构建多层溯源记录 / Build multi-layer provenance entries
entries = [
    build_provenance_entry("Studio A", "create", "Original artwork", layer=0),
    build_provenance_entry("Editor B", "modify", "Color correction", layer=1),
    build_provenance_entry("Publisher C", "distribute", "Final release", layer=2),
]

# 嵌入到不同 DCT 系数层 / Embed into different DCT coefficient layers
watermarked = multi_layer_embed(image, entries, strength=1.0)

# 提取所有溯源层 / Extract all provenance layers
chain = multi_layer_extract(watermarked)
for entry in chain.entries:
    print(f"Layer {entry.layer}: {entry.message}")
```

### 零知识证明 / ZK Proof

```python
from stegmark.core.zk_proof import embed_with_zk, prove_ownership, verify_zk_proof

# 嵌入水印并生成 ZK 承诺 / Embed watermark and generate ZK commitment
watermarked, commitment = embed_with_zk(image, engine, "Alice 2026")

# 证明所有权 / Prove ownership
proof = prove_ownership(watermarked, engine, "Alice 2026", commitment)

# 验证证明（无需原始图像）/ Verify proof (no original image needed)
is_valid = verify_zk_proof(proof)
print(f"Ownership verified: {is_valid}")
```

### AIGC 溯源 / AIGC Provenance

```python
from stegmark.core.aigc_fingerprint import AIGCMetadata, stamp_image, extract_aigc_metadata

# 创建 AIGC 元数据 / Create AIGC metadata
metadata = AIGCMetadata(
    generator="StableDiffusion v3",
    model_version="3.5",
    seed=42,
    prompt_hash="a1b2c3d4",
)

# 嵌入到图像 / Stamp into image
stamped = stamp_image(image, engine, metadata)

# 提取 AIGC 元数据 / Extract AIGC metadata
extracted = extract_aigc_metadata(stamped, engine)
print(extracted.generator, extracted.model_version)

# 导出 C2PA 兼容清单 / Export C2PA-compatible manifest
manifest = metadata.to_c2pa_manifest()
```

### 感知自适应嵌入 / Perceptual Adaptive Embedding

```python
from stegmark.core.perceptual_strength import compute_jnd_map, adaptive_delta

# 计算 JND 图（高纹理区=高值，平坦区=低值）/ Compute JND map
jnd = compute_jnd_map(y_channel)

# 根据 JND 自适应调整嵌入强度 / Adaptively adjust embedding strength
deltas = adaptive_delta(jnd, bits, base_delta=12.0, strength=1.0)
```

### 取证分析 / Forensic Analysis

```python
from stegmark.evaluation.forensics import generate_full_report

# 生成 4 张取证报告图 / Generate 4 forensic report images
files = generate_full_report("original.png", "watermarked.png", "./forensics_output")
# 输出: diff_heatmap, dct_modifications, frequency_analysis, robustness_heatmap
```

### 对抗训练 / Adversarial Training

```python
from stegmark.training.adversarial import adversarial_training_step

# 单步对抗训练 / Single adversarial training step
losses = adversarial_training_step(
    encoder, decoder, cover, message,
    epsilon=0.03, adv_steps=5,
    image_loss_fn=mse_loss, message_loss_fn=bce_loss,
    lambda_adv=0.3,
)
print(f"Image loss: {losses['image_loss']}, Adv msg loss: {losses['adv_message_loss']}")
```

---

## 引擎说明 / Engine Notes

- `auto` 默认解析为 `native` 引擎，确保离线可预测行为。
  `auto` resolves to `native` by default for predictable offline behavior.
- `trustmark` 需显式启用，依赖外部 `trustmark` 包。
  `trustmark` is an explicit opt-in backend requiring the external `trustmark` package.
- `trustmark` 文本模式仅支持 ASCII，默认 BCH-5 配置下最多 8 个字符。
  `trustmark` text mode accepts ASCII only, up to 8 characters under the default BCH-5 profile.
- `hidden` 包含 PyTorch 模型定义、数据集工具、训练脚本、ONNX 导出和运行时适配器。
  `hidden` includes PyTorch model definitions, dataset helpers, trainer scaffold, ONNX export, and runtime adapter.
- `hidden` 运行时需要 `~/.stegmark/models/hidden/` 下的 `encoder.onnx` 和 `decoder.onnx` 文件。
  `hidden` runtime requires exported `encoder.onnx` and `decoder.onnx` files under `~/.stegmark/models/hidden/`.
- `reversible` 使用红色通道 LSB 嵌入，支持通过 `restore()` 完全恢复原始图像。
  `reversible` uses red-channel LSB embedding, supporting full image restoration via `restore()`.
- `screen_resilient` 在 FFT 幅度谱的对数极坐标中频环带用 QIM 嵌入，天然抗旋转和缩放。
  `screen_resilient` uses QIM embedding in the log-polar FFT magnitude spectrum, inherently resilient to rotation and scaling.
- 所有引擎在缺少依赖时抛出 `EngineUnavailableError`，不会静默回退。
  All engines raise `EngineUnavailableError` when dependencies are missing, never silently fall back.

---

## Hidden 工作流 / Hidden Workflow

### 导出 ONNX 模型 / Export ONNX Models

```bash
python scripts/export_onnx.py --output-dir ./.artifacts/hidden
```

### 训练 / Training

```bash
python scripts/train_hidden.py --config configs/hidden_v02.yaml --dataset ./images
```

---

## 基准测试门限 / Benchmark Gates

```bash
python scripts/check_benchmark_gate.py --report benchmark-report/benchmark.json --min-average-bit-accuracy 0.9
```

CI 工作流 / CI Workflows:
- `.github/workflows/ci.yml` — Ruff + Mypy + Pytest + Build
- `.github/workflows/benchmark.yml` — 基准测试 / Benchmark

---

## 可用攻击列表 / Available Attacks

| 攻击名称 / Attack Name | 说明 / Description |
|---|---|
| `jpeg_q90` / `jpeg_q75` / `jpeg_q50` | JPEG 压缩（质量 90/75/50）/ JPEG compression |
| `resize_0.75` / `resize_0.5` | 缩放后恢复 / Resize and restore |
| `crop_0.75` / `crop_0.5` | 裁剪后恢复 / Crop and restore |
| `gaussian_blur_1` / `gaussian_blur_2` | 高斯模糊（半径 1/2）/ Gaussian blur |
| `gaussian_noise_0.03` | 高斯噪声（σ=0.03）/ Gaussian noise |
| `brightness_1.3` | 亮度调整（×1.3）/ Brightness adjustment |
| `dropout_0.1` | 随机像素丢弃（10%）/ Random pixel dropout |

---

## 项目结构 / Project Structure

```
stegmark/
├── cli.py                    # CLI 入口 / CLI entry point
├── config.py                 # 配置管理 / Configuration management
├── types.py                  # 类型定义 / Type definitions
├── exceptions.py             # 异常类 / Exception classes
├── core/
│   ├── engine.py             # 引擎基类 / Engine base class
│   ├── native.py             # DCT 频域引擎 / DCT frequency engine
│   ├── hidden.py             # HiDDeN ONNX 引擎 / HiDDeN ONNX engine
│   ├── reversible.py         # 可逆 LSB 引擎 / Reversible LSB engine
│   ├── screen_resilient.py   # 抗翻拍 FFT 引擎 / Screen-resilient FFT engine
│   ├── trustmark.py          # TrustMark 适配器 / TrustMark adapter
│   ├── provenance.py         # 多层溯源链 / Multi-layer provenance chain
│   ├── zk_proof.py           # 零知识证明 / Zero-knowledge proof
│   ├── perceptual_strength.py # JND 自适应嵌入 / JND adaptive embedding
│   ├── aigc_fingerprint.py   # AIGC 溯源指纹 / AIGC provenance fingerprint
│   ├── codec.py              # 编解码器 / Codec
│   ├── registry.py           # 引擎注册表 / Engine registry
│   └── image_io.py           # 图像 I/O / Image I/O
├── evaluation/
│   ├── benchmark.py          # 基准测试 / Benchmark
│   ├── attacks.py            # 攻击模拟 / Attack simulation
│   ├── metrics.py            # 质量度量 / Quality metrics
│   ├── forensics.py          # 取证分析 / Forensic analysis
│   └── gates.py              # 质量门限 / Quality gates
├── training/
│   ├── adversarial.py        # 对抗训练 / Adversarial training
│   ├── trainer.py            # 训练器 / Trainer
│   ├── losses.py             # 损失函数 / Loss functions
│   └── export.py             # ONNX 导出 / ONNX export
└── service/                  # 服务层 / Service layer
```

---

## 许可证 / License

Apache License 2.0
