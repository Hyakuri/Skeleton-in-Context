# SARS 可见语义 Anchor 适配器实施计划

> **执行者要求：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，按任务逐项执行并更新复选框状态。

**目标：** 使用单一、确定性的配对语义 anchor 策略替换 root-only 对齐，使其支持论文 `bottom` mask。

**架构：** 在窗口切分前构建一组 F64 query/prompt anchor，保留现有尺度、关节和轴契约，四个 F16 窗口共享同一逆变换状态。公开 transform mode 保持不变，并在结果 metadata 中记录实际 anchor。

**技术栈：** Python 3.7 兼容语法、NumPy、PyTorch test doubles、unittest。

## 全局约束

- SiC 测试只使用独立 `SkeletonInContext` 环境。
- 未单独确认前不运行真实 GPU 推理。
- 保留用户未提交的 direct-run 路径修改。
- 不修改 SARS-Inter 生产代码。
- 不改变显式 mask 和 observed-joint 恢复策略。

### 任务 1：回归测试

**文件：**
- 修改：`tests/test_sars_coordinate_adapter.py`
- 修改：`tests/test_sars_completion_adapter.py`
- 修改：`tests/test_sars_completion_series.py`

- [x] 新增项目 bottom 必须选择 upper-torso anchor 的测试。
- [x] 新增 root-compatible 数值等价与全不可见拒绝测试。
- [x] 新增完整 completion 测试，证明 bottom root 仍为 missing，且可见点精确恢复。
- [x] 运行聚焦测试，确认新 bottom 测试在当前 root-only 实现中按预期失败。

### 任务 2：配对 anchor 实现

**文件：**
- 修改：`sars_adapter/coordinate_adapter.py`

- [x] 实现确定性的 anchor 候选和共享 F64 插值。
- [x] 保存通用 query/prompt anchor 状态与 metadata。
- [x] 保留现有 transform mode 标识和 inverse API。
- [x] 运行聚焦测试，确认新增场景全部通过。

### 任务 3：审计与文档

**文件：**
- 修改：`sars_adapter/run_completion.py`
- 修改：`docs/SARS_Adapter_Inference_Guide.md`
- 修改：`docs/SARS_Adapter_Inference_Guide_CN.md`
- 修改：`docs/SARS_Fair_Completion_Pipeline_Guide.md`
- 修改：`docs/SARS_Fair_Completion_Pipeline_Guide_CN.md`

- [x] 捕获 Git identity stderr，不改变 identity bytes 和失败处理。
- [x] 说明 anchor 层级、OOD 边界和可见点精确恢复契约。
- [x] 运行 py_compile、完整 SARS adapter unittest 和 `git diff --check`。
- [x] 审查暂存范围只包含 SiC adapter 相关文件。
