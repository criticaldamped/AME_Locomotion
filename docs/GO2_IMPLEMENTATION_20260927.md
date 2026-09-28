> 历史记录：2026-09-28 已改为正式粗糙地形 Stage1 → 复杂地形 Stage2。本文旧训练阶段及命令不再作为正式流程；控制、观测与原始验证记录供追溯。当前操作请看 [GO2_AME.md](GO2_AME.md)。

# Go2 AME：实现、验证与操作说明

验证日期：2026-09-27。**实现与有界数值检查通过；测试策略尚未学会行走，更未验证长期收敛或实机部署。** 正式训练未启动。最终结果位于 `diagnostics/go2_20260927/`；`suite/` 是奖励调整前的对照，`final_suite/` 是最终版本。

## 环境与已有内容保护

- 项目：`/home/robot1/lgc/AME_Locomotion-main`，本地源码副本，没有 `.git`，没有发现适用的 AGENTS.md。
- 指定环境：`/home/robot1/miniforge3/envs/isaaclab23-ame`，Python 3.11.15、PyTorch 2.7.0+cu128、Isaac Sim 5.1.0、Isaac Lab 本地源码版本 2.3.2（发行包 isaaclab 0.54.2）、RSL-RL 发行元数据 3.0.1、Gymnasium 1.2.1、NumPy 1.26.0。
- Isaac Lab 实际导入目录是 `/home/robot1/IsaacLab-2.3.2/source/`。专用入口显式使用本项目 `rsl_rl` 和 `source/ame_locomotion`，避免错误导入环境中同名包。不安装、不升级依赖，不修改任何 Conda 环境。
- GPU：RTX 5090，32607 MiB。开始时 PID 46969 的其他 Go2 训练占用约 9116 MiB，未停止、修改该进程或其文件。测试与其共享 GPU，吞吐不是独占 GPU 基准。
- 上游 [AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion/tree/1d3519ee946c846c1c89dd660539a224f99972b5) 的比较记录保存在 `upstream_comparison.json`。用户已有 `scripts/rsl_rl/play.py` 与该上游不同，完整保留。
- 修改前的 Python 校验值：`baseline_sha256.json`；公共 PPO 原文件备份：`original/rsl_rl/rsl_rl/algorithms/ppo.py`。已有模型、日志、attention 文件均未覆盖。

## 文件与设计

| 文件 | 用途 |
|---|---|
| `unitree_model/Go2/SOURCE.json` | 固定资产来源、提交、逐文件 SHA-256 |
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/env_cfg.py` | 基础、地形微调、平地和分级台阶配置 |
| 同目录 `mdp.py` | 动作限位、坐标变换、命令区分奖励、地形课程 |
| 同目录 `runner_cfg.py`、`__init__.py` | PPO 参数和独立任务注册 |
| `rsl_rl/rsl_rl/modules/go2_actor_critic.py` | 复用 AME CNN＋多头注意力，Go2 的归一化和有限值检查 |
| `rsl_rl/rsl_rl/algorithms/ppo.py` | 默认关闭的有限值检查开关、可配置学习率边界 |
| `scripts/go2/run.py`、`runner.py` | 诊断、训练、恢复、确定性播放和检查点契约 |
| `scripts/go2/export.py` | TorchScript / ONNX 导出及输出对齐 |
| `scripts/go2/mdp_checks.py`、`bounded_suite.py` | 奖励／课程逻辑检查与有界集成验证 |
| `tests/test_go2_network.py` | 网络、梯度、导出和 G1 回归测试 |

资产来自 [Unitree 官方 unitree_model](https://github.com/unitreerobotics/unitree_model/tree/b6a8942b0803b6c137e58cef12beb4b03e4a2fa7/Go2)，固定提交 `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7`。下载了真正的 Git LFS 内容，约 18.8 MB，包含主 USD、base/physics/sensor 三个依赖层及原许可证；USD 依赖加载检查通过，没有借用其他本地项目中来源不明的模型。

### 控制和动作

沿用本机 Isaac Lab 2.3.2 官方 Go2 基线的 **DCMotor** 模型，P=25、D=0.5，力矩上限 23.5 N·m、速度上限 30 rad/s，按其转速—力矩包络限幅，并显式设置仿真限制。没有未经校准套用本项目另一套 Go2HV 非线性曲线。仿真步长 0.005 s、每 4 步更新动作（50 Hz）。保留基线的关闭自碰撞设置；这是仿真建模取舍，不能据此推断实机无自碰撞风险。

**显式执行器中 P/D 生效；PhysX 的关节驱动 P/D 为 0 是正常现象。** `robot_contract.json` 的 `explicit_actuators` 才是有效 PD 参数；早期诊断文件的 `kp/kd` 是 PhysX 字段，后续入口把它们标为 `physx_kp/physx_kd`。实际执行器参数已读回验证为 25/0.5。

动作／关节观测顺序都显式固定为 SDK 顺序，不能直接用 USD 的天然顺序：

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf   （名称均带 _joint 后缀）
```

默认姿态：右 hip −0.1、左 hip +0.1，前 thigh 0.8、后 thigh 1.0、全部 calf −1.5 rad；初始机身 z=0.4 m。模型落地后的 PD 平衡姿态不等于生成时的离地姿态。

目标 `q_target = q_default + 0.25 × raw_action`，动作缩放取自 Isaac Lab Go2 基线。目标限位由 USD 实际硬限位中央 90% 推导，**没有统一的 ±1、±4 动作裁剪**：

| 关节类别 | 硬限位（rad，约） | 执行目标软限位（rad，约） |
|---|---|---|
| hip | −1.0472 到 1.0472 | −0.94248 到 0.94248 |
| 前 thigh | −1.5708 到 3.4907 | −1.31772 到 3.23762 |
| 后 thigh | −0.5236 到 4.5379 | −0.27052 到 4.28482 |
| calf | −2.7227 到 −0.83776 | −2.62845 到 −0.93201 |

每个关节对应的原始动作边界 `(q_soft_limit-q_default)/0.25` 在实际仿真读回后保存。默认姿态必须位于软限位之内。探针还注入 ±1000 的原始动作到**预处理器**检查投影，但没有把这些极端动作送入物理仿真；真正物理测试使用零动作和 ±0.1 的小幅 thigh 动作。

策略分布是 latent Gaussian，PPO 存储和 log-prob 都针对未经裁剪的原始动作；环境只投影位置目标。`last_action` 和动作变化惩罚同样使用这个原始动作，环境 wrapper 不另行限幅。这是显式定义的带确定性执行映射的 MDP，不把裁剪后的值冒充高斯采样值。另有超限量惩罚和整段 rollout 裁剪比例监测；熵仍是 latent 分布熵，不能解释为实际关节目标熵。

足端精确匹配 FR/FL/RR/RL_foot（带完整原名），四足接触历史参与滑动惩罚，thigh/calf 非期望接触受罚。机身接触或倾斜超过 1 rad 终止；20 秒为超时重置。接触传感器覆盖所有机器人刚体，各项通过 body_names 选取，地形扫描仅射向 `/World/ground`。

### 观测和网络

- 策略：角速度 3＋重力投影 3＋命令 3＋相对关节位置 12＋关节速度 12＋上次原始动作 12＝45；随后拼接 XYZ 地图 2079，总 **2124**。
- 价值网络多真实机身线速度 3，总 **2127**。关节位置／速度观测与动作使用相同显式顺序。
- 地图覆盖 x∈[−0.8,0.8]、y∈[−0.5,0.5] m，分辨率 0.05 m，33×21 点。射线起点高于机身 2 m，向下射线。
- **地图坐标原点在机身，而不是高处射线起点**，仅消除 yaw，+x 前、+y 左、+z 上。保留真实相对高度，不把正高度强制截成 0。
- `ordering='xy'`，展平时 x 最快变化，网络 reshape 为 `[B,21,33,3]`；每步实际检查 x/y 邻接间隔和历史动作一致性。地图始终位于观测尾部。
- 无效射线或任何非有限观测直接报错，生成 FAILED 记录并非零退出。Go2 路径不做 `nan_to_num`。
- 复用原 AME 的 XYZ CNN＋16 头、64 维注意力，保留下采样；Actor/Critic 隐层为 256/128/64。输出 12 维 Gaussian 均值和标量价值。
- Go2 用 **GroupNorm(4)** 替换 CNN BatchNorm，消除 minibatch、单机器人播放及 train/eval 模式之间的统计差异。未改动 G1 的 BatchNorm。父类构造日志会先打印原模块，Go2 最终模块和检查点实际为 GroupNorm。
- 固定物理缩放（角速度×0.2、关节速度×0.05），经验归一化关闭；禁止误开启原实现中不适配地图拼接的经验归一化开关。
- 训练采样，播放用高斯**均值**、关闭观测噪声和随机站立、不采样动作；可重复性不等于 GPU 物理逐位确定性。

### 奖励、课程与随机化

移动命令的平移奖励减去静止基线；非零平移命令的偏航奖励按实际沿命令方向运动比例门控。静止拒绝移动时不会赚取这两项任务奖励。真正的零命令仍奖励静止，并启用姿态保持惩罚。保留速度跟踪、四足腾空、足滑、力矩、加速度、动作变化、关节限位及终止惩罚。腾空奖励只在移动命令有效。没有 survival 常数奖励；奖励正值不能作为成功行走证据。

基础阶段：平地、10% 站立命令，前进 0.2–1.0 m/s、横向 ±0.2 m/s、偏航 ±0.5 rad/s。微调阶段：上／下台阶和小幅粗糙地形，6 级，台阶 0.02–0.14 m，前进 0.3–0.8 m/s、零横移／偏航，固定朝向以让位移课程含义明确。评测固定非零速度并关闭随机站立。

微调引入保守的摩擦随机化（静摩擦 0.6–1.0、动摩擦 0.5–0.6）和本体观测噪声。基础阶段和评测采用固定摩擦。**未开启质量/COM/推力随机化、地图失效/延迟模型**，避免把未经测试的强扰动混入初始 Go2 任务；这些属于进一步鲁棒性及实机迁移验证范围。

课程只在 episode 重置时更新：非零前进命令、存活超过 5 s、实际 +x 位移超过指令位移 65% 且超过 2 m 才升级；运动不足或跌倒降级，站立不升级，最高级封顶。初始站位、横向漂移、跌倒后位移均不会误算升级。已用张量测试覆盖这些边界；短训练没有自然推进到高等级，因此课程长期行为尚未验证。

评测 `--level 0/1/2/3` 固定地形行，分别在约 2–4/4–6/6–8/8–10 cm 范围取阶高。评测使用金字塔台阶、机器人在中央平台，因此首先面对**下台阶**；不代表已通过上台阶、跨沟或整条路线。平台缩小到 1.2 m，让初始扫描就能看到台阶。完整微调训练另包含反向台阶用于上台阶学习。

### 数值与检查点

Go2 使用 log-std，初始 std=0.5、熵系数 0.003；std 超出 [0.02,2.0] 会报错停止，**不通过静默夹紧掩盖膨胀**。PPO 逐项检查观测、动作、log-prob、价值、奖励、returns、advantages、KL、ratio、损失、梯度和更新后参数。梯度范数非有限时在 optimizer step 前停止。学习率自适应限制在 1e−5 到 3e−4，梯度裁剪 1.0。

检查点保存网络、Adam 状态、学习率、下一轮编号、Python/NumPy/Torch/CUDA RNG、全局步数及各环境地形等级；包含观测／关节／动作／模型源码校验契约。架构、地图、关键源代码不同会拒绝加载，G1 预训练模型不可直接用作 Go2 模型。实际用 ame1.pt 启动 Go2 的负例检查已得到契约拒绝和退出码 1，确认异常不会被 Isaac Sim 的关闭流程吞掉。

- `--resume`：相同阶段、相同 num_envs、相同 seed 的续训，恢复上述训练状态和地形等级。
- `--warm_start`：只加载网络，包括同形状的 AME 权重；新建优化器、学习率、迭代、课程及环境状态。用于基础→微调或改变批量规模。
- 物理位置/速度、接触历史、episode 进度、当前命令、部分随机化与事件计时、rollout 缓冲、TensorBoard 累积时间会重新初始化。因此续训**不是逐步完全复现不中断轨迹**。
- `--iterations N` 总是额外 N 轮；文件 `model_11.pt` 表示刚完成第 12 轮，内部 next iteration=12，恢复不会重复最后一轮。
- 新运行目录拒绝覆盖；检查点在该新目录中原子写入。正常每 100 轮和结束保存；异常或 Ctrl-C 后从最近完整检查点恢复，可能丢失最近不足 100 轮的进展。
- 导出模型输入 `[B,2124]`、输出原始动作 `[B,12]`；不包含传感器预处理、PD 或目标投影。使用导出模型时必须同时保留 `robot_contract.json` 与环境参数。这不是实机部署包。

## 实际验证

8 项 CPU 测试通过，包含 G1 的两份已有预训练检查点严格加载、G1 PPO 与修改前实现使用相同随机种子的逐参数精确回归、训练/播放和 batch 大小一致性、非有限输入/梯度/损失注入阻断、TorchScript/ONNX 与原策略对齐。独立配置检查覆盖全部 6 个任务、资产引用、USD 关节限位，以及奖励和课程边界。

最终物理测试：8 环境、250 步（每环境 5 s）的重置／零动作／小幅受控动作，无跌倒、无重置、无动作裁剪和非有限值；显式 PD、力矩上限、动作目标限位及地图排列断言通过。

最终 PPO：64 环境×12 轮，18432 环境步，16.87 s；随后恢复 2 轮。地形 warm-start 2 轮，并另外恢复 2 轮以检查课程状态加载。512 环境×4 轮只作资源测量。包含调整前对照，总计 42 个分散的测试 PPO 迭代，单次最多 12 轮；没有执行收敛训练。

以下评测均使用最终 12 轮测试检查点、16 环境×500 步（每环境 10 s），固定 `(vx,vy,wz)=(0.5,0,0)`，站立概率=0，均值推理：

| 场景 | 实际平均 vx (m/s) | 平均前向位移 (m) | 跌倒/重置 | 原始动作绝对最大值 | 裁剪比例 | 非有限值 |
|---|---:|---:|---:|---:|---:|---:|
| 平地 | −0.00171 | −0.01494 | 0/0 | 0.1811 | 0 | 0 |
| 台阶 level 0 | −0.00169 | −0.01478 | 0/0 | 0.1817 | 0 | 0 |
| 台阶 level 2 | −0.00169 | −0.01476 | 0/0 | 0.1823 | 0 | 0 |

另有 16 环境×1100 步（22 s）的平地测试：16 次预期超时重置、0 次跌倒、0 个非有限值，确认自动重置与时间截断链路。地形恢复记录显示下一轮编号 2、学习率 0.0003、terrain_levels_restored=true。

std 为约 0.488–0.509。短训练所有 rollout 的原始动作最大绝对值 2.313，裁剪比例 0。台阶地图高差实际约 0.022–0.035 m / 0.067–0.077 m，平地差异在浮点误差范围。奖励各分项、实际 vx/vy/wz、逐环境位移、等级等已保存在 `summary.json` 和各目录 `result.json`、`steps.jsonl`。位移排除重置瞬间的跳变；平均速度包含落地瞬态。

**这些策略基本停留在出生点，尚未沿台阶前进。** 无跌倒仅说明在本次短场景中未触发跌倒，不能解释为地形通过率。没有完成长期训练、多种速度／方向／扰动、多随机种子的策略收敛验证；强扰动和实机适配均未验证。

G1：训练和播放任务配置加载通过，两个环境 20 个仿真步通过，原网络和已有 ame1/ame2 检查点推理通过；公共 PPO 默认路径与原实现逐参数一致。没有运行 G1 正式训练或完整摄像机播放场景。

### 资源建议

最终 512 环境的 4 轮实测约 **5407–7435 环境步/s**（含采样和更新），共 7.96 s，不含应用启动。PyTorch 峰值 allocated 3.71 GB、reserved 4.21 GB；整机显存采样峰值 **16526 MiB**，包括 Isaac Sim/PhysX、桌面及另一个 9116 MiB 训练进程。不能把 Torch 数字当作完整仿真进程显存。

建议本机正式训练先用 **512 环境**；调试用 8–64。未实测 1024/2048/4096，因此不建议直接按显存余量外推并发数。其他 GPU 任务和地形规模变化都会改变吞吐；独占时可以自行先做 4 轮的更大规模测试，再决定是否增加。

## 直接运行

以下都在同一个终端依次执行。不要使用原 G1 的 `run_train.sh/run_play.sh` 启动 Go2；Go2 使用专用入口连接检查型 runner。

```bash
source /home/robot1/miniforge3/etc/profile.d/conda.sh
conda activate isaaclab23-ame
cd /home/robot1/lgc/AME_Locomotion-main
nvidia-smi
```

第一阶段基础训练（正式长训练仅供你自行执行）：

```bash
BASE_RUN="logs/rsl_rl/go2_ame/base_$(date +%Y%m%d_%H%M%S)"
python scripts/go2/run.py --mode train --task AME-Go2-Base-v0 \
  --headless --num_envs 512 --iterations 3000 --seed 42 --output "$BASE_RUN"
BASE_CKPT="$(find "$BASE_RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)"
```

同阶段续训；若重新开了终端，请把 BASE_CKPT 设为控制台打印的完整检查点路径：

```bash
python scripts/go2/run.py --mode train --task AME-Go2-Base-v0 \
  --headless --num_envs 512 --iterations 1000 --seed 42 \
  --resume --checkpoint "$BASE_CKPT"
```

确定性平地评测；每环境 22 s，包含 20 s 超时重置：

```bash
python scripts/go2/run.py --mode play --task AME-Go2-Flat-Play-v0 \
  --headless --num_envs 16 --steps 1100 --velocity 0.5 --seed 42 \
  --checkpoint "$BASE_CKPT" --export
```

有界的**当前测试模型**可直接播放（没有学会行走）；去掉 `--headless` 打开可视化窗口：

```bash
python scripts/go2/run.py --mode play --task AME-Go2-Flat-Play-v0 \
  --num_envs 4 --steps 1000 --velocity 0.5 \
  --checkpoint /home/robot1/lgc/AME_Locomotion-main/diagnostics/go2_20260927/final_train64/model_11.pt
```

进入第二阶段前，建议先用多个 seed、0.3/0.5/0.8 m/s 分别进行至少 20 s 的非零命令评测；速度误差应大部分时间小于约 20%，位移与指令累计距离相称，无持续拖脚、频繁跌倒或动作饱和。至少观察数百轮趋势，不能仅凭单次正奖励切换。本次测试模型**不满足**这个进入条件。

第二阶段地形微调，不修改 G1 的 FINETUNE：

```bash
ROUGH_RUN="logs/rsl_rl/go2_ame/rough_$(date +%Y%m%d_%H%M%S)"
python scripts/go2/run.py --mode train --task AME-Go2-Rough-v0 \
  --headless --num_envs 512 --iterations 2000 --seed 42 \
  --warm_start --checkpoint "$BASE_CKPT" --output "$ROUGH_RUN"
ROUGH_CKPT="$(find "$ROUGH_RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)"
```

微调续训和固定难度评测：

```bash
python scripts/go2/run.py --mode train --task AME-Go2-Rough-v0 \
  --headless --num_envs 512 --iterations 1000 --seed 42 \
  --resume --checkpoint "$ROUGH_CKPT"

python scripts/go2/run.py --mode play --task AME-Go2-Obstacle-Play-v0 \
  --headless --num_envs 16 --steps 1100 --level 0 --velocity 0.5 \
  --checkpoint "$ROUGH_CKPT"
# 将 --level 依次改为 1、2、3；去掉 --headless 可视化。
```

上游的两阶段通过 FINETUNE 切换**粗糙基础地形→更复杂的沟隙/桩等地形及奖励/随机化**，但开关本身不负责加载检查点。Go2 沿用“基础策略→地形微调”的流程，特意采用**平地基础→四足分级台阶/粗糙地形**，并显式 warm-start；两阶段网络输入和动作契约一致。这不等同于复刻 G1 的全部地形难度。

## 可重复诊断和停止条件

```bash
python tests/test_go2_network.py
python scripts/go2/run.py --mode config --headless
python scripts/go2/run.py --mode probe --headless --num_envs 8 --steps 250 --velocity 0
python scripts/go2/run.py --mode train --task AME-Go2-Base-v0 \
  --headless --num_envs 64 --iterations 12
# 输出目录默认自动带微秒时间戳，不覆盖已有结果。
```

有界套件（含 2+2+4 轮训练、固定评测和 G1 回归），必须提供现有**最终 Go2**检查点和一个新目录：

```bash
python scripts/go2/bounded_suite.py \
  --root "diagnostics/recheck_$(date +%Y%m%d_%H%M%S)" \
  --checkpoint /home/robot1/lgc/AME_Locomotion-main/diagnostics/go2_20260927/final_train64/model_11.pt
```

默认训练日志在 `logs/rsl_rl/go2_ame/`；显式 --output 以指定目录为准。监控 `iterations.jsonl` 和 TensorBoard 中 Loss、Diagnostics、Episode_Reward、Episode_Termination、Curriculum。刚启动且没有完整 episode 时 Episode_Reward 可能仍为零；应结合 rollout 和评测数据判断。

停止／回退建议（经验排查阈值，不是已验证的收敛保证）：

- 任何非有限值、无效射线、关节／模型契约不符立即停止，查 FAILED 和终端错误，从此前完整检查点修复后重启。不要用 NaN 替换继续跑。
- 裁剪比例连续约 50 轮高于 5%，或突然超过 20%；原始动作幅度持续增加但位移不增加：停止检查动作语义、关节目标及奖励，不先放宽限位。
- std 持续升至 1 以上并继续增大且跟踪不改善：检查熵权重和饱和；超过 2 或低于 0.02 自动停止。不要只看总体 std 均值，需看每关节 min/max。
- 学习率已在下限仍反复出现大梯度／损失跃迁：停止定位传感器、重置和奖励异常。可同时观察 `last_gradient_norm`，它记录裁剪前范数。
- 非零命令评测实际前向速度长期低于命令 20%、位移不随时间增加，且连续约 300–500 轮没有改善：暂停检查奖励分项／足滑／动作噪声，不把站稳或正奖励当作成功。
- 课程均值下降伴随跌倒增加或跟踪退化，持续约 100 轮：保存证据、停止当前微调，评测较早检查点；不要人为升高地形等级掩盖退化。
- 出现 OOM 或整机显存接近上限、影响已有工作：停止当前 Go2 进程，下次减少环境数。改变环境数采用 --warm_start，会重新初始化 optimizer/课程；普通 --resume 要求环境数一致。

正式长训练由用户自行启动；当前交付只证明实现与有限测试范围内的数值链路，不能承诺可靠收敛。
