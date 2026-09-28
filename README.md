# AME Locomotion Reproduction (Isaac Lab + Unitree G1)

Language: [中文](#中文) | [English](#english)

服务器不能使用 Git 时，参见 [浏览器下载 ZIP 与服务器安装说明](docs/SERVER_DOWNLOAD.md)。

---

## 中文

### 项目简介

本项目是对论文 Attention-Based Map Encoding for Learning Generalized Legged Locomotion 中
基于注意力的地形编码器方法 AME 的复现实现。

- 仿真与训练平台: NVIDIA Isaac Sim 5.1.0 + Isaac Lab 2.3.0
- 机器人平台: Unitree G1 29DoF
- 强化学习框架: RSL-RL (含本项目自定义网络扩展)

核心目标是基于高程图和注意力机制，学习具备更强地形泛化能力的腿式运动策略。

### 方法实现位置

AME 的主要网络实现在:

- [rsl_rl/rsl_rl/modules/actor_critic_encoder.py](rsl_rl/rsl_rl/modules/actor_critic_encoder.py)

该文件包含:

- 地形图卷积特征提取
- 局部地形特征与本体状态的多头注意力融合
- 与 Actor/Critic 网络对接的编码输出

### 安装说明

1. 安装 Isaac Lab

- 参考官方文档: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- 建议使用 conda 环境。

2. 安装本项目扩展

```bash
python -m pip install -e source/ame_locomotion
```

3. 安装自定义 rsl_rl

```bash
python -m pip install -e rsl_rl
```

说明: 本项目使用了自定义 Actor-Critic 地形编码网络，若不安装本地 rsl_rl，可能无法正确导入或运行。

4. 预训练模型目录

- 项目内提供了预训练模型目录: [pretrained/](pretrained/)
- 本仓库已提供预训练检查点文件（.pt），可直接用于快速测试。
- ame1.pt: 当前默认配置训练得到（高程图 33×21，开启 CNN 下采样）。
- ame2.pt: 在默认配置基础上加入全局上下文（attach_global=True）。

### 快速开始

训练:

```bash
bash run_train.sh
```

测试与可视化:

```bash
bash run_play.sh
```

### 两阶段训练说明

请注意 AME 采用两阶段训练流程:
第一阶段完成后，将 [velocity_env_cfg_29dof.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py) 中的 `FINETUNE` 设为 `True`，再启动第二阶段训练。

### 复现说明与实现调整

整体上我们遵循论文设计，同时尝试了三点小调整。

1. CNN 输入使用 xyz 坐标，而非仅 z 高度

- CNN 直接处理 xyz 三维输入。
- CNN 输出后不再额外拼接坐标。
- 我们观察到该设置通常有更好的训练表现。
- 可能原因是: 直接输入 xyz 能让网络更早学习到位置相关信息。

2. 用 CNN 步长下采样降低注意力计算开销

- 提高高程图分辨率或扩大范围时，MHA 序列长度会显著增加。
- 我们在 CNN 阶段通过步长下采样缩短后续注意力序列。
- 测试中未观察到明显的最终策略性能下降，同时训练开销更低。

3. 引入 AME2 中的全局上下文

- 我们加入了 AME2 提出的通过 MLP + max-pool 获取全局上下文信息的设计。
- 测试发现该设计能够提升策略表现。
- 代价是注意力权重的可解释性变差，并且训练开销明显增加。

### 关键文件

- AME 编码器实现: [rsl_rl/rsl_rl/modules/actor_critic_encoder.py](rsl_rl/rsl_rl/modules/actor_critic_encoder.py)
- 训练脚本: [run_train.sh](run_train.sh)
- 测试脚本: [run_play.sh](run_play.sh)

---

## English

### Overview

This repository reproduces the Attention-Based Map Encoding (AME) method from the paper
Attention-Based Map Encoding for Learning Generalized Legged Locomotion.

- Simulation and training stack: NVIDIA Isaac Sim 5.0.0 + Isaac Lab 2.3.0
- Robot platform: Unitree G1 29DoF
- RL stack: RSL-RL (with custom network extensions in this project)

The goal is to learn robust legged locomotion policies with stronger terrain generalization using
elevation-map observations and attention-based terrain encoding.

### Where AME Is Implemented

The core AME network implementation is in:

- [rsl_rl/rsl_rl/modules/actor_critic_encoder.py](rsl_rl/rsl_rl/modules/actor_critic_encoder.py)

This file contains:

- CNN-based terrain feature extraction
- Multi-head attention fusion between local terrain features and proprioceptive state
- Encoded outputs for the Actor/Critic heads

### Installation

1. Install Isaac Lab

- Follow the official guide: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- Conda is recommended.

2. Install this extension package

```bash
python -m pip install -e source/ame_locomotion
```

3. Install the local custom rsl_rl

```bash
python -m pip install -e rsl_rl
```

Note: This project depends on a customized Actor-Critic terrain encoder. Without installing the
local rsl_rl package, imports and runtime behavior may fail.

4. Pretrained checkpoint folder

- A pretrained checkpoint folder is included: [pretrained/](pretrained/)
- This repository includes released pretrained checkpoint files (.pt) for quick testing.
- ame1.pt: trained with the current default setup (33x21 elevation map, CNN downsampling enabled).
- ame2.pt: default setup plus global-context (attach_global=True).

### Quick Start

Train:

```bash
bash run_train.sh
```

Play / evaluate:

```bash
bash run_play.sh
```

### Two-Stage Training Note

Please note that AME uses a two-stage training pipeline:

1. Finish stage-1 training first.
2. After stage-1 is done, set `FINETUNE = True` in [velocity_env_cfg_29dof.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py).
3. Then start stage-2 training.

### Reproduction Notes and Small Deviations

Our implementation mostly follows the paper design, with three small adjustments.

1. CNN input uses xyz coordinates instead of z-only height

- The CNN consumes full xyz map coordinates.
- We remove the post-CNN coordinate concatenation.
- In our tests, this often gives better training results.
- A likely reason is that position-related cues are learned earlier and more naturally.

2. CNN stride-based downsampling to reduce attention cost

- Higher map resolution or larger map coverage increases MHA sequence length.
- We apply stride-based downsampling in the CNN stage to shorten the sequence before MHA.
- Empirically, we did not observe clear final-policy degradation, while training cost is reduced.

3. AME2-style global-context

- We also add the AME2 design that extracts global context with an MLP + max-pool pathway.
- In our tests, this improves policy performance.
- The trade-off is weaker attention-weight interpretability and noticeably higher training cost.

### Key Files

- AME encoder: [rsl_rl/rsl_rl/modules/actor_critic_encoder.py](rsl_rl/rsl_rl/modules/actor_critic_encoder.py)
- Training script: [run_train.sh](run_train.sh)
- Play script: [run_play.sh](run_play.sh)
- Pretrained folder: [pretrained/](pretrained/)


## Go2 AME（独立任务）

新增 Go2 本体＋XYZ 高度图 AME 训练、两阶段衔接、确定性评测与诊断。参见 [Go2 实现、验证与操作说明](docs/GO2_AME.md)。短训练测试模型尚未学会可靠行走，正式训练需自行执行。原 G1 任务和脚本保留。

2026-09-28：Go2 正式两阶段已修正为 `AME-Go2-Stage1-v0`（基础粗糙地形）→ `AME-Go2-Stage2-v0`（桩、桥、沟隙等复杂地形）。平地只用于诊断/评测。请以 [最新 Go2 操作说明](docs/GO2_AME.md) 为准。
