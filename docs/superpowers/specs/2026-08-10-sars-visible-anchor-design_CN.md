# SARS 可见语义 Anchor 坐标适配设计

## 目标

让现有正式 SARS H36M17 坐标适配器能够处理 `bottom` 这类 F64 全程缺失
pelvis 与双髋的遮挡，同时不修改显式 missing mask，也不读取 clean 或评价侧数据。

## 范围

- 只修改 Skeleton-in-Context 的 SARS adapter、测试和 adapter Guide。
- 保留现有公开 transform-mode 标识，避免破坏 SARS-Inter run plan 与硬编码配置接口。
- 直接替换旧 root-only 实现，不保留可切换的新旧实现。
- 不修改 SiC 网络、checkpoint、prompt 选择、SARS-Inter 识别、导出、导入和评价行为。

## 坐标契约

每个 F64 sample 按固定顺序选择一种语义 anchor：

1. pelvis，并允许逐帧使用双髋中点；
2. center torso joint 7；
3. upper torso joint 8；
4. joints 11 与 14 的双肩中点；
5. 至少三个可观测关节组成的固定有序集合质心。

query anchor 只读取 `masked_keypoint` 中由显式 `missing_mask` 判定为可见的坐标。
anchor 缺帧使用时间插值，序列边界使用最近观测延伸。prompt anchor 从 train-only
demonstration input 读取完全相同的语义关节。四个不重叠 F16 窗口共享同一条 F64
anchor 轨迹和同一个尺度比。

适配器不会在模型推理前填充任何缺失 query joint。query 与 demonstration input 继续
使用同一显式 mask。逆变换后只接收 `missing_mask=True` 的坐标，并精确恢复全部原始
可见项目坐标。

## 安全与追踪

- anchor 选择不能读取 clean query、类别标签、HAR 结果或 test-derived prompt。
- 整条序列完全不可见或可见骨段不足时仍明确失败。
- 每个 sample 记录 anchor mode、语义 joint IDs、观测与插值支持数量、query/prompt
  anchor hash、尺度比和可见点精确恢复误差。
- 官方 MC 兼容性继续单独报告；坐标适配成功不代表 `bottom` 变成 MC 分布内遮挡。

## 验收

- 项目 `bottom` joints 0-7 全帧缺失时使用 upper-torso anchor，并在不改变 mask 的
  情况下进入 SiC 推理。
- 原 root policy 已支持的输入保持数值等价的 canonical 坐标和逆变换。
- 合成正逆变换误差小于 `1e-6`。
- 全部关节不可见时仍拒绝。
- SARS-Inter 源码不发生修改。
