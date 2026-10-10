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

IK 有两种模式（构造模型时选择）：

| 模式 | 构造 | 公开 q（`model.nq`） |
| --- | --- | --- |
| 仅手臂（默认） | `FairinoQPModel()` | 14 维：左臂 `j1..j7`、右臂 `j8..j14` |
| 手臂 + 躯干 | `FairinoQPModel(active_torso=True)` | 19 维：左臂 7、右臂 7、躯干 `j20 j19 j18 j16 j15` |

两种模式下 `q[:7]`、`q[7:14]` 含义相同；`model.split_q(q)` 返回 `(left, right, torso)`。
头部和轮子始终锁定：

- `IKOptions.torso_position`：仅手臂模式下躯干锁定的 5 个角度（弧度），默认全 0（直立）；
  手臂 + 躯干模式下躯干取自 q，此项被忽略；
- `IKOptions.head_position`：`j17` 头部偏航（弧度）。

躯干角度会改变手臂安装位置，因此 FK、IK、`full_q` 和 MeshCat 显示都使用同一组躯干值。
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
  `IKOptions.default_velocity_limit`（手臂，默认 π rad/s）和 `torso_velocity_limit`
  （躯干，默认 0.5 rad/s），仅对 URDF 速度不为正的关节生效。
- URDF 网格路径从 `package://fairino_wheel_legged_robot_description/meshes/` 改为相对路径
  `meshes/`，其余内容与原文件一致。
- URDF 中 `j11`（右臂肘）的上限为 `2.9548`，而左臂对应的 `j4` 为 `1.9548`，疑似导出笔误；
  本项目按原值使用，未做修改。

## 手臂 + 躯干 IK

`FairinoQPModel(active_torso=True)` 让 IK 同时求解躯干 5 轴：腿部 `j20/j19/j18`
（三根轴都平行于 y，在矢状面内负责躯干高度、前后位置和俯仰）、侧倾 `j16`、偏航 `j15`。
目标仍是两只手的世界位姿，躯干只是多出来的自由度，用来扩大可达范围（下蹲、前探、转身）。

求解策略是**手臂优先**，分两段：

1. 先固定躯干，只用手臂迭代最多 `arms_first_iterations` 次（默认 50；
   实测手臂可达目标的收敛次数中位数 17、最大 31）。手臂够得到时，躯干**完全不动**；
2. 手臂够不到时，从第一段的结果继续，放开全部 19 个自由度迭代，直到总次数达到 `max_iterations`。

躯干很重，连续跟踪时这样可以让躯干尽量保持不动。设 `arms_first_iterations=0` 则从一开始就
联合求解（更快，但小幅移动时躯干也会跟着动几度）。`torso_step_weight`（默认 0）可以在联合阶段
额外惩罚躯干步长，但在手臂接近奇异时会明显拖慢收敛，一般不需要。

实测（手臂在身前的初始位形）：仅手臂模式下，双手前移 0.55 m，或下移 0.5–0.7 m，都会失败；
手臂 + 躯干模式能收敛（约 125–170 次迭代，2 ms 左右），躯干主要靠 `j19/j18` 下蹲前倾。
100 组随机可达 19 维目标全部收敛。

注意：

- 不会自动回到直立：手臂优先只保证"能不动就不动"，躯干离开直立后不会被拉回；
  需要时用 `posture_target`/`posture_weight`，或在 Qt 中拖动躯干滑条；
- **没有重心/稳定性约束**：躯干前倾或下蹲时重心会前移，实机使用前需要另加重心约束，
  并确认 URDF 中的质量（SolidWorks 估算值）；
- 腿部 `j20/j19` 在直立时就处在下限 0，所以直立姿态本身在关节限位上；
- 只检查手臂与躯干/底盘的碰撞，不检查手臂与腿部连杆的碰撞。

## 家位姿态（零空间姿态任务）

7 轴手臂做 6 维末端任务，每条臂有 1 个冗余自由度；躯干参与时再多 5 个。只靠"从上一帧解出发"，
冗余自由度会随路径漂移（实测躯干模式绕一圈后关节偏离 110°）。设一个家位姿态 `posture_target`
并令 `posture_weight > 0` 后：

- 冗余自由度在末端任务的**零空间**里被拉向家位，末端位姿不受影响（无残差偏置）；
- 每次求解仍从 seed（上一帧结果）出发，保持连续；
- 目标在家位末端附近时，解就落在家位关节角上。实测从扰动 15° 左右的初值出发，回到家位的
  关节误差 < 1e-4 rad；沿 0.1 m 的圈跟踪一周，最大偏离 16°，回到起点后与家位一致；
- 每次求解迭代到姿态步长 < `posture_tolerance`（默认 1e-4 rad）为止；
  `result.posture_residual` 是剩余的零空间姿态偏差，被限位/避碰挡住时不为 0；
- 躯干模式下，启用姿态任务后，手臂到位以后还会放开躯干，把躯干也拉回家位（手保持不动）。

```python
home = np.radians([-30, 45, 0, 60, 0, 0, 0,  30, 45, 0, 60, 0, 0, 0,  30, 60, 30, 0, 0])
model = FairinoQPModel(active_torso=True)
options = IKOptions(posture_weight=1.0, posture_target=home)
result = model.solve_ik(left_target, right_target, previous_q, options)
```

Qt 默认把初始关节值作为家位（`--posture-weight 0` 关闭），IK 页有 "Set home"
和 "Go home" 两个按钮；`--home-q` 用度给出家位（14 个，或含躯干的 19 个），同时作为初始位形：

```bash
fairino-ik-qt --active-torso    # 默认家位就是上面的初始位形
fairino-ik-qt --active-torso --home-q 0 45 0 90 0 45 0  0 45 0 90 0 45 0  0 0 0 0 0
```

## 底盘（vx, vy, rz），不参与 IK

底盘在单独的 `fairino_ik.chassis` 模块里，IK 求解器完全不涉及底盘：IK 只解手臂（和躯干），
目标都在机器人坐标系下。底盘控制器只输出车体坐标系的速度
`(vx, vy, rz)`（m/s、m/s、rad/s，x 前、y 左），舵轮逆解交给底盘控制器。

| 接口 | 作用 |
| --- | --- |
| `ChassisController(ChassisLimits(...))` | 速度限幅（三轴等比例缩放，保持方向）+ 加速度限幅 |
| `controller.teleop((vx, vy, rz), dt)` | 摇杆/上层给的速度经限幅后输出 |
| `controller.command(odom_pose, goal_pose, dt)` | 按 SE(2) 误差的比例控制开向目标位姿，`reached` 表示到达 |
| `controller.stop(dt)` | 按加速度限制减速到 0 |
| `integrate_twist(pose, twist, dt)` / `body_twist(p0, p1, dt)` | SE(2) 精确积分 / 反算速度 |
| `world_to_robot(T, odom_pose)` / `robot_to_world(T, odom_pose)` | 手的目标在里程计坐标系和机器人坐标系之间换算 |
| `base_goal_for_hands(left_w, right_w, left_ref, right_ref)` | 求一个底盘位姿，使两只手的世界目标落在参考手位（如家位 FK）上 |

默认限幅 0.5 m/s、0.5 m/s、0.5 rad/s，加速度 1 m/s²、1 m/s²、1 rad/s²，需按实车修改。

**机械臂基座在哪**：IK 的目标始终相对 `base_link`（x 前、y 左、z 上）。`base_link` 和
躯干、双臂都装在底盘上，跟着底盘一起移动。所以底盘行驶时手臂和躯干关节不动，
`base_link` 下的 IK 目标也不变，IK 不需要重新求解。

```python
controller = ChassisController()
while running:
    cmd = controller.teleop(joystick_twist, dt)   # 或 controller.command(odom, goal, dt)
    send_chassis_velocity(*cmd.as_tuple())        # 只发给底盘，与 IK 无关
    # 手臂/躯干照常：目标在 base_link 下
    result = model.solve_ik(left_target, right_target, q, options)
```

只有当上层希望手停在世界里某个固定位置时，才需要用 `world_to_robot(T_world, odom)`
把世界目标换算成 `base_link` 目标再交给 IK；这是可选的，默认不做。

Qt：`fairino-ik-qt --chassis`。底盘面板有 vx/vy/rz 滑条、Stop chassis、Reset odometry，
显示限幅后的速度指令和积分得到的里程计位姿；MeshCat 里整车（含 `base_link`、躯干、手臂）
一起移动，手臂和躯干关节保持不变。IK 页的目标始终是 `base_link` 坐标。
`--chassis-max-velocity`、`--chassis-max-acceleration` 设置限幅。

Qt 默认就是手臂 + 躯干联动求解（19 维）；`--no-active-torso` 只解手臂、躯干锁定在 `--torso`。

## 躯干避碰

与 tron2-ik 的中心立柱避碰对应，这里检查手臂和机身之间的碰撞：

- 障碍物：`body_Link`（网格包围盒，截到 z=0.45 m，不含颈部以上）、`waist_Link` 和
  底盘 `base_link` 的网格包围盒；前两个随躯干关节一起移动。躯干参与 IK 时，
  约束使用手臂采样点相对盒子的相对速度，躯干可以主动让开；
- 采样：每条手臂肩→肘→腕心→法兰三段，每段 4 个采样点加法兰端点，按 `body_link_radius`
  （默认 0.06 m）膨胀成近似胶囊；
- 在 QP 中作为线性化最小间距约束 `body_min_distance`（默认 0.01 m），激活距离
  `body_activation_distance`（默认 0.04 m）。

零位时肩关节到躯干盒的固定间距为 0.0295 m，因此结果里的 `body_nearest_segment`
在手臂远离躯干时通常显示肩部。目标落在障碍物内部时，IK 会停在安全间距上（容差 10 µm）
并返回 `success=False`；躯干参与 IK 时，原本在直立躯干内部的目标可能因躯干让开而变得可达。
用 `IKOptions(body_avoidance=False)` 或 `--no-body-avoidance` 关闭。

## 安装

推荐新建独立环境：

```bash
cd /home/xense/fairion/fairino_ik
mamba env create -f environment.yml
mamba activate fairinoik
```

or

```bash
cd /home/xense/fairion/fairino_ik
mamba create -n fairino python=3.12
mamba env update -n fairino -f environment.yml
mamba deactivate
mamba activate fairino
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
python examples/dual_arm_qt.py --no-active-torso  # 只解手臂，躯干锁定（默认手臂+躯干联动）
```

界面（约 850×390 像素）：顶部一行是 MeshCat 地址、Open MeshCat、QP 后端和 Reset；
下面是 `FK joints`、`IK targets` 两个标签页，`--chassis` 时多一个 `Chassis` 页。
数值框右侧单独显示单位（deg、m、m/s、rad/s），各页的使用说明在标签页的悬停提示里。

- FK 页：左臂、右臂、躯干/头部三列关节滑动条；
- 躯干 5 轴 + 头部滑动条（改动后手臂关节不变、IK 目标同步到新的 FK 位姿）。
  `--active-torso` 时躯干滑条就是 q 的一部分，会随 IK 结果更新；
- 左右末端各自的 XYZ 和旋转目标滑动条。旋转滑条 `rot x/y/z` 是绕世界固定轴、
  相对上次 FK 同步姿态的旋转向量（度），没有万向锁；绝对 RPY 只做显示
  （在 pitch=±90°，即工具竖直时，RPY 本身是奇异的）；
- 默认初始位形（也是默认家位）：左臂 `[-30, 45, 0, 60, 0, 0, 0]`°、右臂 `[30, 45, 0, 60, 0, 0, 0]`°、
  躯干 `j20 j19 j18 j16 j15 = [30, 60, 30, 0, 0]`°。仅手臂模式下躯干锁定在这个角度。
  可用 `--home-q`/`--initial-q`/`--torso` 覆盖。全零位是双臂伸直的奇异位形
  （j1/j3/j5 共线，雅可比秩只有 4），不适合作为拖动起点；
- IK 页按钮：`Solve left`、`Solve right`、`Solve both`、`Sync from FK`、`Set home`、`Go home`；
- 求解状态两行：结果、迭代次数（及躯干角度），位置/姿态误差与避碰间距。

求单臂时，另一侧目标会保持当前 FK 位姿，因此仍由同一个双臂 QP 连续求解；
躯干参与时，躯干移动后另一只手会自动补偿以保持原位。

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

双手前下方的目标超出手臂可达范围，需要躯干下蹲。不加 `--active-torso` 时失败（误差 0.22 m），
加上后收敛：

```bash
python examples/dual_arm_ik.py --active-torso \
  --left-target 0.7507 0.2354 0.5038 0.2618 0 1.5708 \
  --right-target 0.7507 -0.2354 0.5038 -0.2618 0 -1.5708 \
  --q 0 45 0 90 0 45 0  0 45 0 90 0 45 0
```

`--active-torso` 时 `--torso` 是躯干初值；`--arms-first-iterations`、`--torso-step-weight`
对应同名选项。

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

手臂 + 躯干：

```python
model = FairinoQPModel(active_torso=True)
q_seed = np.concatenate((np.radians([0, 45, 0, 90, 0, 45, 0] * 2), np.zeros(5)))
left_target, right_target = model.end_effector_poses(q_seed)
left_target[:3, 3] += [0.2, 0.0, -0.5]
right_target[:3, 3] += [0.2, 0.0, -0.5]
result = model.solve_ik(left_target, right_target, q_seed)
left_q, right_q, torso_q = model.split_q(result.q)
```

独立目标可通过批量接口减少 Python/C++ 往返；左右目标形状为
`(B, 4, 4)`，初值形状为 `(B, nq)`，返回按输入顺序排列的 `IKResult` 列表：

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

测试覆盖模型资源、关节分组、14/19/32 维映射、世界坐标系朝向、躯干锁定、法兰帧、双臂 FK→C++ QP IK、
手臂优先的躯干 IK、躯干/底盘避碰、批量接口、独立于 IK 的底盘 (vx, vy, rz) 控制以及 Qt 离屏构造。

## 目录

```text
src/fairino_ik/                 Python 包
src/fairino_ik/assets/fairino/  可随 wheel 安装的 URDF/STL
models/fairino/                 指向包内模型的源码树快捷路径
cpp/                            C++ 求解器、pybind11 绑定和 QP 单元测试
examples/                       CLI 与 Qt 启动脚本
tests/                          自动测试
```
