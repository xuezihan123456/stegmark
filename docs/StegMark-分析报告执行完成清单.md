# StegMark 分析报告执行完成清单

基于 `StegMark-全面分析与优化报告.md` 的落实结果。

生成日期：2026-04-12
代码目录：`.worktrees/v0-1`

## MUST FIX

- [x] 修复 CLI 中 `strength`/`workers` 的 `or` 误判，显式 `0.0` 和 `0` 不再被吞掉
- [x] 修复 `bits_hex` 带 `0x` 前缀时 `bytes.fromhex(...)` 崩溃
- [x] 为图像读取增加文件大小上限与像素上限防护
- [x] 为批处理 `workers` 增加安全上限
- [x] 为输出路径增加边界检查，防止写出允许目录
- [x] native DCT 路径改为批量块处理，去掉逐块 Python 循环
- [x] 为以上关键修复补充回归测试

## SHOULD FIX

- [x] HiddenEngine 懒加载 ONNX session 加锁，避免并发重复初始化
- [x] `ImageMetadata.extras` 改为真正不可变
- [x] 抽出共享 payload 解析，消除 native/hidden 的重复逻辑
- [x] benchmark 改用 metrics 中的 PSNR 计算，消除重复实现
- [x] hidden 训练网络里的重复 `_conv_block` 提取到共享层
- [x] 全局 logging 基础设施
- [x] config 写入改为安全序列化，支持特殊字符转义
- [x] 收紧关键依赖下界：`numpy`、`Pillow`、`onnxruntime`
- [x] registry 改为缓存实例 + 延迟导入 + entry points 预留

## NICE TO HAVE

- [x] benchmark 拆分为 `benchmark.py` / `types.py` / `reports.py`
- [x] WatermarkEngine 增加能力声明/内省属性
- [x] 配置系统支持引擎级别嵌套，补齐 hidden 配置与 providers
- [x] 批处理增加进度回调
- [x] CLI 的 `SystemExit` 路径切到 Click 友好的退出方式
- [x] 批处理对 native/auto 支持进程池，并在不适用时回退
- [x] codec 位处理向量化
- [x] benchmark 攻击并行化
- [x] 攻击模拟支持可选 `seed`
- [x] 改善 `EmbedResult` / `StegMark` / metadata 的 repr 可读性
- [x] 为引擎注册预留插件机制
- [x] `EmbedResult.save()` 从数据类中完全移除
- [x] 批处理后主动释放图像数据的额外内存回收策略
- [x] benchmark/业务层更深一步的边界重构
- [x] 全面性能基准前后对比报告

## 额外完成

- [x] 新增 benchmark 并行测试
- [x] 新增 image IO 安全测试
- [x] 新增报告回归测试：CLI/service、core、benchmark、config/hidden
- [x] runtime info 暴露 hidden provider 策略
- [x] hidden provider 优先级与 CPU 回退逻辑

## 验证

- [x] `python -m py_compile` 覆盖关键模块通过
- [x] 自定义 `tmp_path` harness 下整体验证：`120 passed`
- [x] benchmark 相关测试集验证：`27 passed`

## 清理状态

- [x] 本次主线程自己创建的 `_tmp_runtime` 已清理
- [ ] 若干历史/子代理临时目录仍残留：当前沙箱 ACL 拒绝访问，无法无提权删除

涉及目录示例：

- `.pytest-base`
- `.pytest-base-python`
- `.pytest-base-python2`
- `.pytest-tmp`
- `.pytest_tmp`
- `.tmp-worker-c`
- `.tmp-worker-e`
- `tests/.pytest_tmp`
- `tests/.tmp_api`
- `tests/_pt3` / `_pt4` / `_pt5` / `_pt6`

这些目录不属于功能改动本身；若需要，可在具备权限的本地 shell 中统一清理。
