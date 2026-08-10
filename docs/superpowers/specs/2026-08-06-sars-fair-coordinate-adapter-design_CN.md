# SARS 公平坐标适配器设计

## 目标

在不泄露评价信息、不修改源数据集的前提下，使用冻结的 Skeleton-in-Context checkpoint，对 SARS-Inter 自获数据集和 NW-UCLA 进行客观的骨骼补值比较。

## 隔离边界

- SiC 只读取 `completion_query.pkl`、冻结 checkpoint、对应 source config，以及官方 `3DPW_MC/train` demonstration bank。
- SiC 不读取 evaluation sidecar、标签、clean skeleton、识别分数或源数据集。
- SiC 只新建 `completion_result.pkl`，不覆盖 query、checkpoint、源数据集或历史结果。
- SARS-Inter 继续负责结果导入、可见点精确恢复、识别、统计和可视化。

## 标准转换

SARS 项目的 H36M17 是左腿优先、右臂优先；SiC/MotionBERT 是右腿优先、左臂优先。正向置换为：

```text
sic_joint[j] = project_joint[[0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13][j]]
```

该置换自身即为逆置换。项目坐标契约为 `(x_lateral, y_depth, z_height)`，转换到 SiC Y-up 空间的固定右手系旋转为：

```text
(x, y, z) -> (x, z, -y)
```

逆变换为 `(x', y', z') -> (x', -z', y')`。

## Root 与尺度对齐

每个 F64 sample 只计算一次 root 和 scale，四个 F16 窗口共享。

- Root 轨迹优先使用可见 pelvis；pelvis 缺失时使用可见双髋中点；其余缺口仅根据可用 root anchor 做线性插值，序列边界使用最近 anchor 延伸。
- Sample scale 使用完整 F64 中两端均可见的 H36M 骨段长度中位数。
- 每个 sample 从 `3DPW_MC/train` 确定性选择一个 demonstration，四个窗口重复使用同一个 demonstration。
- Reference scale 使用相同可见骨段支持，在重复到 F64 的 demonstration 上计算。
- Query 被平移和缩放到该 demonstration 的坐标框架；转换后再次把缺失坐标严格置零。
- 模型输出经过逆变换恢复到项目关节顺序和坐标轴；只接收缺失位置，原项目空间中的可见坐标精确恢复。

若整条序列没有 root anchor，或可见骨段不足，适配器必须明确失败。不得使用数据集专用常数、clean pose、标签、关节插值补值、平滑或骨长修正。

## 窗口策略

F64 划分为 `[0:16]`、`[16:32]`、`[32:48]` 和 `[48:64]`。四个窗口共享同一个 demonstration、sample scale、root policy、关节置换和轴旋转。继续保留与当前官方 SiC 输入长度一致的非重叠推理方式；窗口边界的位置和速度跳变只做测量与报告，不静默平滑。

## 可追踪信息

结果 metadata 记录：

- transform mode 与版本；
- 关节置换和轴矩阵；
- sample/reference scale 与可见骨段数量；
- root-anchor policy 和 root 轨迹 hash；
- prompt identity、文件 hash 和选择策略；
- 窗口边界及边界不连续诊断；
- checkpoint SHA256、仓库 commit/worktree identity、运行时间和可见点恢复误差。

## 验收标准

- synthetic 可见输入正逆变换误差小于 `1e-6`。
- joint 与 mask 置换完全一致。
- 进入模型前所有缺失坐标仍严格为零。
- 四个窗口使用同一 prompt identity 和同一 sample scale。
- SiC 无法访问任何私有评价字段。
- 导入后的可见坐标与 query source 位级一致。
- 自获数据和 NW-UCLA 使用相同 checkpoint 与 transform mode。
- 真实 smoke 结果有限、sample-ID 对齐、独立保存，并完成识别和可视化。

