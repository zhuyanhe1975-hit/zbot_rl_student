# zbot_rl_core

这是一个面向 ZBot 模块化机器人的 Isaac Lab + RSL-RL 最小教学仓库。

仓库目标是：方便学生查看任务、播放训练好的结果、继续训练，并在已有任务基础上添加新任务。为了便于学习，常看的入口文件放在浅层目录；可复用实现细节放在内部目录。

## 仓库包含什么

- 6dof 和 8dof ZBot 运动控制任务。
- 双足、双足到蛇形、蛇形、轮式、鸟形、quat 观测、全向速度跟踪等任务变体。
- `6dof-bipedal-velocity-imu` 使用教师-学生机制：教师有仿真特权速度信息，学生只看 IMU 友好的观测。
- `pth/` 中保存了一组已训练好的 checkpoint。
- 全向速度跟踪任务支持键盘控制播放。
- `run.sh` 和 `train.sh` 保持单命令激活形式，方便一眼看清当前运行哪个任务。

## 学生先看哪里

最重要的两个入口文件：

- `zbot_direct/source/zbot_direct/zbot_direct/env.py`
  - 任务环境总入口。
  - 可以看到每个任务使用哪一种 env 行为。
  - 修改 observation 或 env 行为时，从这里开始看。

- `zbot_direct/source/zbot_direct/zbot_direct/cfg.py`
  - 任务配置总入口。
  - 可以对照 robot asset、action/observation 维度、reward scale、command range、termination 参数等差异。
  - 修改 reward 权重或任务参数时，从这里开始看。

常用脚本：

- `run.sh`
  - 播放训练好的策略。
  - 保持只有一条命令处于激活状态。

- `train.sh`
  - 训练策略。
  - 同样保持只有一条命令处于激活状态。

内部实现文件在：

```text
zbot_direct/source/zbot_direct/zbot_direct/tasks/direct/zbot_direct/
```

常见内部文件：

- `base_env.py`：通用 scene setup、reset、reward loop、contact termination、默认 action 积分。
- `bipedal_env.py`：双足任务状态和 observation。
- `transition_env.py`：6dof 双足到蛇形任务逻辑。
- `ground_env.py`：蛇形和轮式任务逻辑。
- `velocity_env.py`：全向速度跟踪主流程。
- `velocity_commands.py`：速度命令采样和键盘/manual command override。
- `velocity_rewards.py`：全向速度跟踪 reward。
- `velocity_debug_vis.py`：速度命令 debug 可视化。
- `shared_rewards.py`：非 velocity 任务共享 reward。
- `zbot_direct_robot_cfgs.py`：机器人 USD 和 articulation 定义。

## 安装与环境检查

```bash
cd /home/yhzhu/myWorks_vips/zbot_rl_student
./install.sh
./list_envs.sh
```

如果 Isaac Lab 不在默认位置，可以这样指定：

```bash
ISAACLAB=/path/to/isaaclab ./install.sh
```

## 播放结果

```bash
./run.sh
```

`run.sh` 的原则是：只保留一条激活命令，其余任务命令都注释掉。这样可以很直观地知道当前要播放哪个任务。

全向速度跟踪任务使用键盘控制，播放时建议 `num_envs=1`：

```bash
$HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play_keyboard.py \
  --task=Zbot-Direct-6dof-bipedal-velocity-v0 \
  --num_envs=1 \
  --real-time \
  --load_run ./pth/Zbot-Direct-6dof-bipedal-velocity-v0
```

键盘控制说明：

- 方向键或小键盘控制 `vx/vy`。
- `Z/X` 控制 yaw 角速度 `wz`。
- `E` 增大速度倍率。
- `Q` 减小速度倍率。
- `L` 清零当前速度命令。

## 训练任务

```bash
./train.sh
```

`train.sh` 和 `run.sh` 使用同样原则：只保留一条激活命令，其余候选任务都注释掉。

训练过程会产生大量临时 checkpoint、TensorBoard 日志、视频和参数文件。默认不要把这些中间结果写到仓库里，而是写到仓库外：

```text
/home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student/
```

`train.sh` 中每条候选命令都显式带有：

```bash
--log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
```

这样训练目录和教学仓库分开管理。仓库里的 `pth/` 只保存少量精选 checkpoint，用于课堂演示、结果复现或 `run.sh` 播放。

## 当前任务 ID

```text
Zbot-Direct-6dof-bipedal-v0
Zbot-Direct-6dof-bipedal-velocity-v0
Zbot-Direct-6dof-bipedal-velocity-quat-v0
Zbot-Direct-6dof-bipedal-velocity-imu-v0
Zbot-Direct-6dof-bipedal-quat-v0
Zbot-Direct-6dof-bipedal-to-snake-v0
Zbot-Direct-6dof-bipedal-to-snake-v1
Zbot-Direct-8dof-bipedal-v0
Zbot-Direct-8dof-bipedal-v1
Zbot-Direct-8dof-bipedal-v2
Zbot-Direct-8dof-bipedal-v3
Zbot-Direct-8dof-bird-v0
Zbot-Direct-8dof-snake-v0
Zbot-Direct-8dof-wheel-v0
Zbot-Direct-8dof-bipedal-velocity-v0
```

## 如何修改已有任务

- 修改 reward 权重：编辑 `zbot_direct/source/zbot_direct/zbot_direct/cfg.py`。
- 修改机器人 asset 或模块连接方式：编辑 `cfg.py`，必要时继续查看 `zbot_direct_robot_cfgs.py`。
- 修改普通任务 observation：先看 `env.py`，再进入 `bipedal_env.py`、`transition_env.py` 或 `ground_env.py`。
- 修改全向速度任务 command 或 reward：查看 `velocity_commands.py` 和 `velocity_rewards.py`。
- 修改 termination：普通任务先看 `base_env.py`，特殊任务再看对应 env 文件。

## 6dof velocity-imu 教师-学生任务

任务 ID：

```text
Zbot-Direct-6dof-bipedal-velocity-imu-v0
```

这个任务用于研究更接近真机部署的观测形式。

- 教师 observation：包含 `base_lin_vel_b` 和 `base_quat_w`，这是仿真特权信息和 IMU 姿态信息，用于生成教师动作。
- 学生 policy observation：不包含 `base_lin_vel_b`，只保留更接近硬件可获得的信息，例如 IMU quat、IMU 角速度、重力方向、关节状态、速度命令和历史动作。
- reward 仍然可以使用仿真里的真实 base 速度，因为 reward 不部署到硬件上。

训练学生前，建议先训练带 quat 的教师任务：

```text
Zbot-Direct-6dof-bipedal-velocity-quat-v0
```

把精选教师 checkpoint 放到：

```text
pth/Zbot-Direct-6dof-bipedal-velocity-quat-v0/
```

然后训练学生，`train.sh` 中已有候选命令：

```bash
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-6dof-bipedal-velocity-imu-v0 --num_envs=1024 --max_iterations=1500 --headless --load_run ./pth/Zbot-Direct-6dof-bipedal-velocity-quat-v0 --log_root_path /home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student
```

训练完成后，把精选 student checkpoint 放到：

```text
pth/Zbot-Direct-6dof-bipedal-velocity-imu-v0/
```

## 如何新增一个任务

新增任务时，优先参考一个相近任务。多数新任务只需要新增一个 env 入口类和一个 cfg 类。

### 1. 在 `env.py` 中添加任务入口

文件位置：

```text
zbot_direct/source/zbot_direct/zbot_direct/env.py
```

选择最接近的父类：

- 双足行走：继承 `ZbotBipedalEnv` 或 `ZbotBipedalJointAccEnv`
- 6dof curriculum 双足：继承 `Zbot6DofBipedalEnv`
- quat observation 任务：继承 `Zbot6DofQuatEnv`
- 双足到蛇形：继承 `Zbot6DofToSnakeEnv`
- 蛇形或轮式：继承 `ZbotSnakeEnv` 或 `ZbotWheelEnv`
- 全向速度跟踪：继承 `ZbotVelocityEnv`

示例：

```python
class Zbot8DofMyTaskEnv(ZbotBipedalEnv):
    """8dof 双足实验任务。"""
```

### 2. 在 `cfg.py` 中添加任务配置

文件位置：

```text
zbot_direct/source/zbot_direct/zbot_direct/cfg.py
```

cfg 里只放这个任务真正不同的内容，例如 robot、action/observation 维度、reward scale、command range、termination 参数。

示例：

```python
@configclass
class Zbot8DofMyTaskCfg(Zbot8DofTaskCfg):
    robot: ArticulationCfg = ZBOT_D_8S_HUMAN_CFG.replace(prim_path=ROBOT_PRIM_PATH)
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 5.0,
            "similar_to_default": -0.2,
            "torques": -0.2,
        },
    }
```

### 3. 在浅层 `__init__.py` 中注册任务

文件位置：

```text
zbot_direct/source/zbot_direct/zbot_direct/__init__.py
```

在 `_TASKS` 里添加一项：

```python
(
    "Zbot-Direct-8dof-my-task-v0",
    "Zbot8DofMyTaskEnv",
    "Zbot8DofMyTaskCfg",
),
```

### 4. 在 `run.sh` 和 `train.sh` 中添加命令

保持仓库原则：只激活一条命令，其余命令注释掉。

训练命令示例：

```bash
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/train.py --task=Zbot-Direct-8dof-my-task-v0 --num_envs=1024 --max_iterations=15000 --headless
```

播放命令示例：

```bash
# $HOME/isaaclab/isaaclab.sh -p zbot_direct/scripts/rsl_rl/play.py --task=Zbot-Direct-8dof-my-task-v0 --num_envs=32 --real-time --load_run ./pth/Zbot-Direct-8dof-my-task-v0
```

### 5. 如果有 checkpoint，放到 `pth/<task-id>/`

示例：

```text
pth/Zbot-Direct-8dof-my-task-v0/model_latest.pt
```

### 6. 检查注册和脚本语法

```bash
./list_envs.sh
bash -n run.sh
bash -n train.sh
```

第一次新增任务时，建议不要急着改深层实现文件。先通过 `env.py` 新增入口、通过 `cfg.py` 新增配置。只有确实需要新增 observation、reward function、command sampler 或 termination rule 时，再进入内部文件。

## 其他说明

- `pth/` 保存已训练 checkpoint。
- 临时训练结果默认保存到 `/home/yhzhu/myWorks_vips/zbot_rl_runs/zbot_rl_student/`，不要提交到 git。
- 旧版 RSL-RL checkpoint 的加载兼容由 `checkpoint_compat.py` 处理。
- 本仓库刻意排除了日志、Hydra 输出、Python cache、build 目录、导出 policy 等自动生成内容，以保持教学仓库简洁。
