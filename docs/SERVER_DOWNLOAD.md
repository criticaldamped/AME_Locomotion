# 通过浏览器下载到服务器

本项目可以从 GitHub 浏览器下载 ZIP，不需要在服务器使用 Git 或 SSH。

## 下载

登录拥有仓库访问权限的 GitHub 账号，打开：

https://github.com/criticaldamped/AME_Locomotion

- 最新代码：点击 **Code → Download ZIP**。包含代码、测试、G1/Go2 机器人资源和原有 G1 预训练模型。
- 本次本机目录快照：进入 **Releases**，下载 `AME_Locomotion-main-snapshot.zip`。额外包含现有诊断结果、短测试检查点、输出和参考文件；排除 Git 内部目录、Python 缓存和安装元数据。
- Release 同时提供 `SHA256SUMS.txt`，用于核对快照下载是否完整。

机器人资源直接存储在仓库中，ZIP 中是实际文件，无需 Git LFS。诊断短测试检查点不代表训练收敛。

## 安装与检查

先在服务器配置与本地兼容的 Isaac Sim、Isaac Lab、PyTorch 和 CUDA 环境。
ZIP 只包含项目文件，不包含 Conda 环境、Isaac Sim 或 Isaac Lab 的安装目录。

解压后打开终端，激活服务器上的 Isaac Lab Python 环境，再进入解压出来的项目目录：

```bash
conda activate <服务器上的IsaacLab环境名>
cd <解压后的项目目录>
python -m pip install -e source/ame_locomotion
python -m pip install -e rsl_rl
python scripts/go2/run.py --mode config --task AME-Go2-Stage1-v0 --headless
```

尖括号内容需要替换为服务器上的实际环境名和路径。项目通过自身所在目录定位机器人资源，无需使用本地 `/home/robot1/...` 路径。

配置检查成功后，按 [Go2 两阶段训练说明](GO2_AME.md) 执行。该说明中开头的 Conda 激活和项目路径也需要替换为服务器路径。
Stage1 到 Stage2 使用 `--warm_start`，同阶段完整续训使用 `--resume`。

后续代码更新可重新下载最新代码 ZIP；训练结果另行保存，更新前结束使用该目录的训练进程。
本地评测模型时保留对应的代码版本、任务阶段和运行配置。
