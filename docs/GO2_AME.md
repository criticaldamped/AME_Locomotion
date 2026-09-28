# Go2 AME：基础粗糙地形 → 复杂稀疏支撑地形

2026-09-28 修正：正式流程与本项目 G1 的两阶段**目标和地形类别**对齐。平地只保留诊断/评测，不需要先训练一个平地收敛策略。之前推荐的“平地→台阶”缩小了 AME 的目标范围，已被以下流程替代。

## 正式任务

| 阶段 | 任务名 | 地形 |
|---|---|---|
| 第一阶段 | `AME-Go2-Stage1-v0` | 上下台阶、上下斜坡、方块、粗糙地面、踏石、小沟隙 |
| 第二阶段 | `AME-Go2-Stage2-v0` | 上下台阶、双列桩、交错桩、断续窄桥、更宽沟隙、横栏、粗糙地面 |
| 第一阶段评测 | `AME-Go2-Stage1-Play-v0` | 同 Stage1，可固定等级、单独选地形 |
| 第二阶段评测 | `AME-Go2-Stage2-Play-v0` | 同 Stage2，可固定等级、单独选地形 |
| 平地诊断 | `AME-Go2-Flat-Play-v0` | 控制、姿态、基本速度跟踪排查 |

旧 `Base`、`Rough`、`Obstacle-Play` 名称保留原含义，用于兼容旧诊断和检查点，**不再代表推荐的正式两阶段**。不要把旧 Rough 当作新的复杂地形 Stage2。

第一阶段从一开始就让 AME 接触非平坦及局部缺失支撑面，学习运动控制与地形编码；第二阶段进一步训练依赖落脚位置选择的稀疏支撑路线。平地测试有助于排错，但在完全平坦地图上无法充分学习这些地形差异。

与 G1 对齐的是阶段逻辑和地形家族；没有宣称照搬 G1 所有尺寸、奖励或逐项复现论文实验。Go2 的腿长、足端、四足支撑范围和驱动参数不同，需要独立调节难度。当前尺寸是待长期训练检验的起始范围，不是已经证明可以通过的性能上限。

## 几何与课程

每阶段 6 行难度、8 列地形，8 个类别等比例分配，避免稀有复杂地形因列数不足而没有生成。地块 8×8 m，高度场横向分辨率 0.05 m、垂直分辨率 0.005 m。新路径配置在：

- `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/stages.py`
- 同目录 `complex_terrains.py`

| 参数 | 简单端 → 困难端 |
|---|---|
| Stage1 台阶高度 | 0.02 → 0.14 m |
| Stage1 踏石宽度 / 间隙 | 0.55 → 0.35 m / 0.05 → 0.12 m |
| Stage1 沟宽 | 0.05 → 0.15 m |
| Stage2 台阶高度 | 0.04 → 0.20 m |
| 双列／交错桩支撑边长 | 0.45 → 0.25 m |
| 桩之间沿前进方向的间隙 | 0.05 → 0.25 m |
| 断续桥纵向支撑长度 | 0.55 → 0.35 m |
| 断续桥横向宽度 | 0.70 → 0.40 m |
| 桥块间隙 | 0.05 → 0.25 m |
| Stage2 沟宽 | 0.10 → 0.35 m |
| 横栏高度 | 0.04 → 0.16 m |

这是连续设计值，实际高度场尺寸会量化到 0.05 m 网格。随机粗糙地形维持对应阶段固定噪声范围，作为基础运动覆盖，不代表所有类别的每个参数都随等级单调变化。

复杂桩/桥/沟隙采用有限深度 −1.2 m 的坑底。坑底是碰撞与射线网格的一部分，扫描返回真实的深负高度；不会把未命中射线或深沟静默变成平地。地图仍相对机器人机身 yaw 坐标系，保留 33×21×3 XYZ、45/48 维本体观测以及原来的 AME CNN/注意力路径。

G1 的部分原生成器存在无种子随机数、沟深硬编码等问题。Go2 独立实现保留相应的桩/桥/同心沟地形结构，使用明确的局部随机种子、可配置坑深和完整出生平台；**G1 原文件未修改**。

新任务始终沿初始 +x 方向训练，平移命令 0.3–0.8 m/s、无侧向/偏航命令、5% 随机站立，出生朝向固定。Stage1 使用固定摩擦和无观测噪声；Stage2 开启已有的本体噪声和摩擦随机化。没有新增未经验证的强推力或质量随机化。

路线课程根据实际 +x 位移更新：存活超过 2 s、前进超过 3 m、横向偏离小于 0.8 m、达到指令累计距离 65% 才可升级；失败或显著运动不足降级，站立不升级，最高级封顶。前进 3.5 m 为路线结束（time_out），避免走入相邻地块；横向偏离超过 1 m 或从后方退出属于失败。

新增按地形类别识别的掉坑终止：桩、桥、沟、踏石上的机身下降到出生参考面下 0.45 m 即判定失败，避免落到人工坑底后继续“走路刷位移”。正常下楼/下坡不应用这个高度阈值。逻辑边界已测试，但长期课程推进和实际复杂落脚策略尚未验证。

## 检查点衔接

两阶段的机器人、关节顺序、动作缩放和限位、本体/地图拼接、GroupNorm、Actor/Critic 形状均不变，因此可以直接迁移网络权重，不需要重新训练或替换 AME 编码器。

- **Stage1 → Stage2 使用 `--warm_start`**：载入整个网络，重置优化器、学习率、迭代、环境和课程。
- 同一阶段使用 `--resume`：恢复网络、优化器、学习率、下一轮编号、RNG、全局步数和地形等级；要求相同 seed、num_envs、训练阶段和地形/奖励/事件/课程配置。
- 新增独立的阶段配置/源码指纹。即便两个阶段都是 terrain generator，也不能把 Stage1 的等级/optimizer 当作 Stage2 状态直接恢复。
- 旧平地/台阶 Go2 检查点形状未破坏，仍可播放或显式 warm-start，但它们不是完成了 Stage1 的证明。
- 物理瞬时状态、episode、当前命令、接触历史和 rollout 缓冲仍会重新初始化；续训不等于逐步复现不中断的物理轨迹。
- `--iterations N` 表示额外 N 轮；所有新运行使用新目录，不覆盖旧模型。

## 操作命令

下列正式长训练没有由助手执行。示例轮数只是预算，不保证收敛。初始建议仍为 512 环境，调试使用 8–64；先检查当前 GPU 上的其他任务。

```bash
source /home/robot1/miniforge3/etc/profile.d/conda.sh
conda activate isaaclab23-ame
cd /home/robot1/lgc/AME_Locomotion-main
nvidia-smi

STAGE1_RUN="$PWD/logs/rsl_rl/go2_ame/stage1_$(date +%Y%m%d_%H%M%S)"
python scripts/go2/run.py --mode train --task AME-Go2-Stage1-v0 \
  --headless --num_envs 512 --iterations 3000 --seed 42 --output "$STAGE1_RUN"
STAGE1_CKPT="$(find "$STAGE1_RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)"
```

在 Stage1 各类基础地形上验证后进入 Stage2，不要求先完成一个独立平地训练。评测应按类别检查路线完成、跌倒、速度/位移和等级分布，避免平均奖励掩盖某类地形不会通过。建议至少在多个 seed、0.3/0.5/0.8 m/s、多个等级上看到稳定路线完成，再开始复杂阶段；例如每类至少 80% 路线完成、低于 5% 跌倒可作为人工审查的起始门槛，并非已经校准的性能标准。本次短测试模型不满足该进入条件。

```bash
python scripts/go2/run.py --mode play --task AME-Go2-Stage1-Play-v0 \
  --headless --num_envs 16 --steps 1100 --terrain stepping_stones --level 1 \
  --velocity 0.5 --checkpoint "$STAGE1_CKPT"

STAGE2_RUN="$PWD/logs/rsl_rl/go2_ame/stage2_$(date +%Y%m%d_%H%M%S)"
python scripts/go2/run.py --mode train --task AME-Go2-Stage2-v0 \
  --headless --num_envs 512 --iterations 3000 --seed 42 \
  --warm_start --checkpoint "$STAGE1_CKPT" --output "$STAGE2_RUN"
STAGE2_CKPT="$(find "$STAGE2_RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)"
```

第二阶段续训、单独评测复杂地形：

```bash
python scripts/go2/run.py --mode train --task AME-Go2-Stage2-v0 \
  --headless --num_envs 512 --iterations 1000 --seed 42 \
  --resume --checkpoint "$STAGE2_CKPT"

python scripts/go2/run.py --mode play --task AME-Go2-Stage2-Play-v0 \
  --headless --num_envs 16 --steps 1100 --terrain bridge --level 3 \
  --velocity 0.5 --checkpoint "$STAGE2_CKPT"
```

`--terrain` 可改为 `double_stakes`、`alternate_stakes`、`bridge`、`gaps`、`rails`、`stairs_up`、`stairs_down`、`rough`；Stage1 另有 `slope_up`、`slope_down`、`boxes`、`stepping_stones`。省略该选项评测混合地形。`--level` 为 0–5，播放/评测关闭课程推进和随机站立；去掉 `--headless` 打开可视化。`--terrain` 仅用于评测/探针，避免误把单一地形当成完整训练分布。

重新开终端后请重新设置检查点变量为实际绝对路径。续训默认写入新的时间戳目录，后续评测如需使用续训结果，应选择该新目录的检查点。

## 有界验证与复查

2026-09-28 的新证据保存在 `diagnostics/go2_complex_20260928/`。验证范围包括 10 个任务的注册/配置、8 项原有网络与 G1 算法/预训练检查点回归，以及地形种子重复性、网格有限性、难度方向、坑深、地形类别覆盖、掉坑分类和课程边界。

```bash
python tests/test_go2_network.py
python scripts/go2/run.py --mode config --task AME-Go2-Stage1-v0 --headless
python scripts/go2/complex_suite.py \
  --output "diagnostics/complex_recheck_$(date +%Y%m%d_%H%M%S)"
```

套件只执行 Stage1 64 环境×4 轮、Stage2 warm-start 4 轮、Stage2 resume 2 轮；随后对双列桩、交错桩、桥和沟隙分别用 8 环境×250 步固定 0.5 m/s 评测，执行 G1 20 步回归，并验证跨阶段错误 --resume 会拒绝。单进程上限 300 s。它不是长期收敛测试。

每次运行保存 env/agent YAML、robot_contract、阶段/地形列清单；训练保存 iterations.jsonl，评测保存 result.json/steps.jsonl，包含速度、位移、动作幅值、裁剪、std、奖励、重置/跌倒、地图高差与非有限值。SUCCESS 只表示程序和断言通过，不表示路线通过。

数值保护沿用此前实现：无效射线/非有限观测、动作、loss、梯度等立即失败，log-std 标准差范围 [0.02,2]，自适应学习率 [1e-5,3e-4]。持续动作饱和、噪声增大、数百轮实际不前进、某类地形长期无法通过或课程持续退化应停止检查。不要用放宽限位、替换 NaN 或强制升高等级掩盖问题。

机器人资产来源、控制/观测契约、PPO 细节及 2026-09-27 的原始证据保存在 [历史实现记录](GO2_IMPLEMENTATION_20260927.md)。其中旧的正式阶段安排已废止，当前任务和命令以本文为准。

## 本次实际结果（2026-09-28）

`diagnostics/go2_complex_20260928/validation_summary.json` 汇总配置、检查点源文件指纹、评测与资源结果；完整逐步日志在 `suite_v2/`。Stage1 4 轮 → Stage2 权重衔接 4 轮 → 同阶段续训 2 轮全部完成，检查点下一轮编号分别为 4、4、6。跨阶段误用 `--resume` 按预期拒绝。10 个任务配置及资产检查、8 项网络测试、G1 两环境 20 步仿真回归通过。本次没有改变 G1 或共享 PPO 文件，既有 Python 文件只修改 Go2 注册、入口和 runner；其余修改是新增文件和文档。

以下均为第 3 级、8 环境、每环境 5 秒、固定前向命令 0.5 m/s、关闭随机站立、确定性均值动作的测试：

| 地形 | 实际前向速度 m/s | 平均前向位移 m | 跌倒／重置 | 动作裁剪／非有限值 |
|---|---:|---:|---:|---:|
| 双列桩 | −0.00902 | −0.04322 | 0 / 0 | 0 / 0 |
| 交错桩 | −0.00948 | −0.04534 | 0 / 0 | 0 / 0 |
| 断续桥 | −0.00937 | −0.04470 | 0 / 0 | 0 / 0 |
| 沟隙 | −0.00876 | −0.04190 | 0 / 0 | 0 / 0 |

地图观测可见约 1.2 m 的地形高差。策略仍停留在出生平台，**没有展示任何复杂路线通过能力**；零跌倒不代表能够过障碍。同一地形的多个环境在关闭随机化后轨迹相同，不能当作独立随机条件的成功率统计。动作幅值、各关节标准差和奖励分项保存在各地形的 `result.json`。

初次 Stage1 启动发现方块网格尺寸 0.4 m 与 8 m 地块整除时触发上游边框网格除零；改为 0.45 m 后完整套件通过。原失败日志保留在 `suite/stage1.log`，没有通过屏蔽异常继续运行。

RTX 5090（32607 MiB）上另执行了 **512 环境、Stage2 仅 2 轮** 的资源测试：约 4532–5435 环境步/秒，两轮采样更新合计 5.28 s，含启动约 14.41 s；PyTorch 峰值 allocated 3.72 GB、reserved 4.21 GB。GPU 全卡占用采样峰值 15888 MiB，包含其他进程，不能视作本任务独占显存。测试期间观察到另一 Conda 环境的训练进程，未对其作任何操作。短窗口吞吐有波动，不能据此保证长训练耗时。

因此建议当前从 **512 环境**开始正式训练；8–64 用于调试。尚未实测本任务的 1024/2048 环境，不建议直接按显存比例扩大。启动前重新检查其他任务占用，运行中若全卡占用持续接近 28 GiB、发生 OOM 或吞吐明显恶化，应停止本次运行并重新评估并发负载。更改环境数量需要显式 warm-start，会重置优化器和课程，不能声称是完整续训。

所有本次有界测试已结束，没有启动正式长训练。硬地形的四足可达性、长期数值稳定性、课程推进、随机条件鲁棒性和策略收敛仍需后续训练与分项评测确认。
