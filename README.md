# Fairino 轮腿双臂 C++ QP IK

这是一个独立的 Fairino 轮腿机器人双臂逆运动学项目，结构参照 `tron2-ik`，只包含：

- `models/fairino/fairino_wheel_legged_robot.urdf` 及其 STL 网格；
- Pinocchio 模型加载、FK 和关节分组映射；
- C++ Pinocchio + eiquadprog 的双末端约束 QP IK；
- 左右臂分别控制的命令行和 Qt 滑动条界面；
- 可选 MeshCat 三维显示。

Python 只提供接口、CLI 和 Qt；FK、雅可比、约束 QP 和迭代全部在 C++ 扩展中执行。

## 关节分组

原始 URDF 由 SolidWorks 导出，关节统一命名为 `j1`..`j28`，名字里没有躯干/左右臂之分。
本项目按运动链确定分组（Python 中为 `JOINT_GROUPS`）：

| 分组 | 关节 | 运动链 |
| --- | --- | --- |
| 左臂 `LEFT_ARM_JOINTS` | `j1 j2 j3 j4 j5 j6 j7` | `left_arm_base_Link` → `left_waist2_Link` |
| 右臂 `RIGHT_ARM_JOINTS` | `j8 j9 j10 j11 j12 j13 j14` | `right_arm_base_Link` → `right_waist2_Link` |
| 躯干 `TORSO_JOINTS` | `j20 j19 j18 j16 j15` | `base_link` → 小腿 → 大腿 → 髋 → 腰 → `body_Link` |
| 头部 `HEAD_JOINT` | `j17` | `body_Link` → `head_Link` |
| 轮子 `WHEEL_JOINTS` | `j21`..`j28` | 4 个转向 + 4 个驱动（continuous） |

注意 URDF 中的 `left/right_waist1/2_Link` 实际是腕部连杆，不是腰。

IK 只求解两条七轴手臂，公开配置向量固定为 14 维：左臂 `j1..j7` 再右臂 `j8..j14`。
躯干、头部和轮子在求解时锁定：

- `IKOptions.torso_position`：按 `j20, j19, j18, j16, j15` 顺序的 5 个角度（弧度），默认全 0（直立）；
- `IKOptions.head_position`：`j17` 头部偏航（弧度）。

躯干角度会改变手臂安装位置，因此 FK、IK、`full_q` 和 MeshCat 显示都使用同一组锁定值。
完整 Pinocchio 配置为 `nq=32, nv=28`（四个 continuous 驱动轮各占 cos/sin 两个坐标）。

## 坐标与末端帧

所有目标位姿都是世界坐标系下的 `4x4` 齐次矩阵。默认世界坐标系是 **x 前、y 左、z 上**，
原点与 `base_link` 重合。

URDF 由 SolidWorks 导出，`base_link` 沿用 CAD 坐标轴：正前方是 −y，左臂在 +x。
本项目把这组坐标轴绕 z 轴顺时针转 90°（从上往下看）作为默认世界坐标系，
即 `DEFAULT_BASE_POSE = Rz(+90°)`。换算关系：`x_world = −y_base`，`y_world = x_base`。
零位时双臂沿 ±y 水平伸直（左臂在 +y）。

需要原始 `base_link` 坐标时，传入 `base_pose=URDF_BASE_POSE`（单位阵），
CLI/Qt 中使用 `--urdf-base-frame`。

默认末端帧为 URDF 最后一节手臂连杆：

- 左臂：`left_waist2_Link`
- 右臂：`right_waist2_Link`

C++ 模型额外加入了法兰帧 `left_flange` / `right_flange`，位于腕部连杆 −x 方向 0.114 m
（由腕部网格的外端推算，接工具前请按实物/手册确认）。需要以法兰为 TCP 时：

```python
model = FairinoQPModel(left_frame="left_flange", right_frame="right_flange")
```

## 与原 URDF 不同的约定

- URDF 中所有关节 `velocity="0"`。QP 每步的速度边界改用
  `IKOptions.default_velocity_limit`（默认 π rad/s），仅对 URDF 速度不为正的关节生效。
- URDF 网格路径从 `package://fairino_wheel_legged_robot_description/meshes/` 改为相对路径
  `meshes/`，其余内容与原文件一致。
- URDF 中 `j11`（右臂肘）的上限为 `2.9548`，而左臂对应的 `j4` 为 `1.9548`，疑似导出笔误；
  本项目按原值使用，未做修改。

## 躯干避碰

与 tron2-ik 的中心立柱避碰对应，这里检查手臂和躯干之间的碰撞：

- 障碍物：`body_Link`（网格包围盒，截到 z=0.45 m，不含颈部以上）和 `waist_Link` 的网格包围盒，
  随锁定的躯干关节一起移动；
- 采样：每条手臂肩→肘→腕心→法兰三段，每段 4 个采样点加法兰端点，按 `body_link_radius`
  （默认 0.06 m）膨胀成近似胶囊；
- 在 QP 中作为线性化最小间距约束 `body_min_distance`（默认 0.01 m），激活距离
  `body_activation_distance`（默认 0.04 m）。

零位时肩关节到躯干盒的固定间距为 0.0295 m，因此结果里的 `body_nearest_segment`
在手臂远离躯干时通常显示肩部。目标落在躯干内部时，IK 会停在安全间距上并返回 `success=False`。
用 `IKOptions(body_avoidance=False)` 或 `--no-body-avoidance` 关闭。

## 安装

推荐新建独立环境：

```bash
cd /home/xense/fairion/fairino_ik
mamba env create -f environment.yml
mamba activate fairinoik
```

`environment.yml` 已经以 editable 模式安装当前项目以及 Qt、MeshCat 和测试依赖。

如果要安装到已有环境：

```bash
python -m pip install -e .
```

该命令会同时安装核心、Qt、MeshCat 和测试依赖，并固定 Pinocchio 3.9.0
与其匹配的 urdfdom/tinyxml2 原生 ABI。请确保 `python` 和 `pip` 来自同一个已激活环境。

启动命令会检测 `/opt/ros` 中不兼容的 Pinocchio Python/动态库路径，并自动以隔离环境重启。也可以手动运行：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH python examples/dual_arm_qt.py
```

QP 后端固定为原生 C++ `eiquadprog`。

## Qt 滑动条界面

```bash
fairino-ik-qt
```

或直接运行：

```bash
python examples/dual_arm_qt.py
python examples/dual_arm_qt.py --no-meshcat
python examples/dual_arm_qt.py --torso 0 0 0 0 0.3 --head 0.2
```

界面包含：

- 左右臂各 7 个关节 FK 滑动条；
- 躯干 5 轴 + 头部的锁定位置滑动条（改动后手臂关节不变、IK 目标同步到新的 FK 位姿）；
- 左右末端各自的 XYZ 和旋转目标滑动条。旋转滑条 `rot x/y/z` 是绕世界固定轴、
  相对上次 FK 同步姿态的旋转向量（度），没有万向锁；绝对 RPY 只做显示
  （在 pitch=±90°，即工具竖直时，RPY 本身是奇异的）；
- 默认初始位形为 `[0, 45, 0, 90, 0, 45, 0]`°（双手在身前）。全零位是双臂伸直的奇异位形
  （j1/j3/j5 共线，雅可比秩只有 4），不适合作为拖动起点；
- `Solve left`、`Solve right` 和 `Solve both`；
- 原生 C++ QP solver 状态与躯干间距；
- Reset、从当前 FK 写入 IK 目标以及动画显示。

求单臂时，另一侧目标会保持当前 FK 位姿，因此仍由同一个 14 维双臂 QP 连续求解。

## 命令行示例

用内置的小幅双臂目标做一次非零 C++ QP 自检：

```bash
fairino-ik
```

指定左右完整 XYZ/RPY 目标（米、弧度）。下面是双臂 `[0, 45, 0, 90, 0, 45, 0]` 度时手在身前的位姿：

```bash
python examples/dual_arm_ik.py \
  --left-target 0.626 0.2354 0.905 -0.7854 0 1.5708 \
  --right-target 0.626 -0.2354 0.905 0.7854 0 -1.5708 \
  --q 0 30 0 60 0 30 0  0 30 0 60 0 30 0
```

查看全部参数：

```bash
fairino-ik --help
fairino-ik-qt --help
```

## Python API

```python
import numpy as np
from fairino_ik import FairinoQPModel, IKOptions

model = FairinoQPModel()
options = IKOptions(torso_position=(0, 0, 0, 0, 0.3), tolerance=1e-4)
q_seed = np.radians([0, 30, 0, 60, 0, 30, 0] * 2)

left_target = model.forward_kinematics(q_seed, side="left", options=options)
right_target = model.forward_kinematics(q_seed, side="right", options=options)
left_target[2, 3] += 0.05
right_target[2, 3] += 0.05

result = model.solve_ik(left_target, right_target, q_seed, options)
print(result.success, result.q, result.as_dict())
```

独立目标可通过批量接口减少 Python/C++ 往返；左右目标形状为
`(B, 4, 4)`，初值形状为 `(B, 14)`，返回按输入顺序排列的 `IKResult` 列表：

```python
results = model.solve_ik_batch(
    np.stack([left_target, left_target]),
    np.stack([right_target, right_target]),
    np.stack([q_seed, q_seed]),
    options,
)
```

每个目标仍使用完整约束 QP，共用一次配置转换和模型锁，不会自动沿用上一目标的解。
`result.full_q` 在同次原生求解中生成，包含躯干、头部和轮子的完整 32 维状态，
可用 `model.joint_coordinates[name]` 查索引。

每轮在 C++ 中解一个带关节位置、速度和躯干线性化安全约束的 QP，再用 Pinocchio 更新状态。
零位是双臂完全伸直的奇异位形，较远或不可达的目标可能返回 `success=False`；
连续拖动时应始终把上一帧结果作为下一帧 seed。

## 测试

```bash
pytest -q
QT_QPA_PLATFORM=offscreen pytest -q tests/test_qt.py
```

测试覆盖模型资源、关节分组、14/32 维映射、世界坐标系朝向、躯干锁定、法兰帧、双臂 FK→C++ QP IK、
躯干避碰、批量接口以及 Qt 离屏构造。

## 目录

```text
src/fairino_ik/                 Python 包
src/fairino_ik/assets/fairino/  可随 wheel 安装的 URDF/STL
models/fairino/                 指向包内模型的源码树快捷路径
cpp/                            C++ 求解器、pybind11 绑定和 QP 单元测试
examples/                       CLI 与 Qt 启动脚本
tests/                          自动测试
```
