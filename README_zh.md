# simtoolreal_unilab

[English](README.md)

在 [UniLab](https://github.com/Motphys/UniLab) + [UniSim](https://github.com/unilabsim/unisim) 上训练 SimToolReal 的 teacher 策略，走 IsaacSim URDF→USD 资产路径：KUKA iiwa14 机械臂（7 自由度）+ SHARPA 手（22 自由度）从 1200 个程序化生成的工具变体池中学习提举与操作，训练语义与原仓库对齐。

这是一个下游任务包：安装后直接使用 UniLab 的共享训练工具链，无需 clone UniLab 本体。物理层运行在 UniSim 的 IsaacSim 子进程后端；学习器是 vendored 且经 SHA 门禁审计的 RL-Games SAPG 运行时。

| 能力 | 状态 |
| --- | --- |
| `SimToolReal` teacher 策略训练（RL-Games SAPG，IsaacSim 后端） | 已完成训练并与原仓库做标签级对拍（见"证据"） |
| 后端支持 | 仅 IsaacSim；MuJoCo 路径未在本仓库实现 |
| 与原 IsaacLab `DirectRLEnv` 实现的标签一致性 | TensorBoard 关键标签与原仓库同 seed 运行不可区分 |
| 学生策略 / 相机 / 接触传感器、部署、sim2real | 未迁移；本仓库不含部署代码 |
| 资产来源 | URDF→USD（机器人/桌子 URDF 随仓库分发，1200 工具池程序化生成）；不支持 MJCF 源 |

## 证据

- **训练对拍（L3）**：12,288 环境、seed 42 的完整训练（514 epochs）对照原仓库前 500 epochs——全部关键 TensorBoard 标签族 `INDISTINGUISHABLE`。
- **逐变体质量读回** vs 生成 manifest：最大相对偏差 5.7e-08。
- **世界系 wrench 通路**：探针 delta 与冻结的 kit 复验锚点逐位一致。
- 验证环境：Linux x86_64、单张 NVIDIA GPU（RTX 4090）、IsaacSim 5.1.0-rc.19 + IsaacLab 2.3.2.post1 作为 worker 运行时。
- 尚未包含机器人视频；一段训练后的提举回合录像本 README 最缺的素材。

## 首次成功运行

要求：Linux x86_64、一张支持 CUDA 的 NVIDIA GPU（学习器与 IsaacSim worker 共用）、[uv](https://docs.astral.sh/uv/)、约 20 GB 磁盘（worker 运行时与缓存）。

```bash
git clone https://github.com/unilabsim/simtoolreal_unilab.git
cd simtoolreal_unilab
uv sync
bash scripts/setup_isaacsim_env.sh   # 一次性：安装 IsaacSim worker 运行时（约 15 GB）
uv run simtoolreal-assets            # 校验随仓库分发的资产
```

以上就是全部安装。两件无法随 `uv sync` 分发的事：

1. **IsaacSim worker 运行时**。UniSim 的 `isaacsim` extra 为空（NVIDIA 的 SDK 不可再分发），worker 跑在独立的 Python 3.11 venv 里。`scripts/setup_isaacsim_env.sh` 一次性把验证过的版本栈装进去（IsaacSim 5.1.0 + IsaacLab 2.3.2.post1 wheel + torch 2.7.0/cu128，来自 NVIDIA PyPI 源（cu128 覆盖 Blackwell 及更早架构）），脚本幂等（分步 marker），装在 `~/.cache/unisim/isaacsim/venv`。**URDF importer 对齐步骤**：pip 轮子捆绑的 urdf importer 扩展是 2.4.30，缺少 IsaacLab 2.3.2+ 所需的 `set_merge_fixed_ignore_inertia` 绑定（全新 pip 安装会在首次 URDF 转换时崩溃）；脚本会从 NVIDIA 扩展注册表的公开 CDN 下载 2.4.31（sha256 钉死）替换之，并在收尾校验该绑定存在——校验失败即安装失败，不会装出必炸的环境。`simtoolreal-train`/`simtoolreal-play` 自动发现该安装——无需配置任何环境变量。想改用别的 IsaacSim 解释器，则 export `UNISIM_ISAACSIM_PYTHON=/path/to/venv/bin/python`（任意 venv 布局；验证时用的就是这条逃生口）。

2. **工具资产**。机器人（KUKA iiwa14 + SHARPA 手）与桌子的 URDF 随仓库分发在 `simtoolreal_unilab/assets/simtoolreal/`（上游 MIT，SHA-256 manifest；`uv run simtoolreal-assets` 校验）。1200 工具池首次使用时程序化生成，缓存在 `/tmp` 下（可用 `SIMTOOLREAL_TOOL_POOL_ROOT` 覆盖；生成的 bootstrap URDF 用 `SIMTOOLREAL_GENERATED_ASSETS_ROOT`）。想改用上游 checkout 的资产，设 `SIMTOOLREAL_ASSETS_ROOT` 指向其 `assets/` 目录。

环境验收 smoke（64 环境、24 工具池、2 epochs——单卡数分钟；这组旗标是必需的，因为生产 rollout 恰好用六个探索块）：

```bash
uv run simtoolreal-train \
  algo.num_envs=64 env.tool_pool_num_per_type=2 \
  rl_games.params.config.max_epochs=2 \
  rl_games.params.config.expl_coef_block_size=32 \
  rl_games.params.config.minibatch_size=256 \
  rl_games.params.config.central_value_config.minibatch_size=256 \
  training.smoke=true training.no_play=true
```

预期输出：逐 epoch 统计，随后在 `logs/rlgames_sapg/SimToolReal/<run>/` 下产出 checkpoint 与 TensorBoard 事件文件。smoke checkpoint 是验收产物，不是策略基线。

## 训练、评估、查看

生产训练使用 owner 默认值（24,576 环境；`num_actors` 必须恰好构成六个 `expl_coef_block_size` 块）：

```bash
uv run simtoolreal-train                 # 默认 task=simtoolreal/isaacsim
uv run simtoolreal-train algo.num_envs=12288 rl_games.params.config.expl_coef_block_size=2048
```

回放最近的 checkpoint 化 run（经 IsaacSim worker 渲染）：

```bash
uv run simtoolreal-play
uv run simtoolreal-play --checkpoint-file logs/rlgames_sapg/SimToolReal/<run>/nn/<checkpoint>.pth
```

列出已注册任务与后端：

```bash
uv run simtoolreal-list-envs            # SimToolReal   isaacsim
```

TensorBoard：`uv run tensorboard --logdir logs/rlgames_sapg`。标签名与原仓库的 tag 集保持兼容，两侧 run 可直接叠加对比。

## 修改与扩展

| 想改什么 | 路径 |
| --- | --- |
| 奖励项与权重 | `src/simtoolreal_unilab/tasks/simtoolreal/rewards.py`，数值在 `conf/rlgames_sapg/task/simtoolreal/base.yaml` |
| 观测（policy/critic 组、行级 reset 路径） | `tasks/simtoolreal/observations.py` |
| 目标/命令逻辑与 success hub | `tasks/simtoolreal/commands.py` |
| 场景组合、工具池接线、资产解析 | `tasks/simtoolreal/scene.py`、`env_cfg.py` |
| 1200 工具池定义与 URDF 生成 | `tasks/simtoolreal/tool_catalog.py`、`tool_urdf_writer.py` |
| 终止与事件（wrench DR、reset） | `tasks/simtoolreal/terminations.py`、`events.py` |
| 算法超参 | `conf/rlgames_sapg/config.yaml` |
| 本任务叠加在 pinned UniLab 修订上的行为 | `src/simtoolreal_unilab/_compat/`（见下） |

## 兼容层（面向开发者）

本任务需要的若干行为上游 UniLab 尚未提供（多刚体根 reset 路由、实体变体质量 facade、manager 日志视图、seed 传播、entity-assets 物化）。`simtoolreal_unilab._compat` 在 import 时把这些方法体移植到 pinned 的官方 UniLab 修订上，pin 漂移会立刻报错。依赖钉定：

- `unilab`——官方 `main` 上本任务开发所基于的确切 commit；
- `unisim-core`——fork 分支 `feat/isaacsim-urdf-multi-entity`，携带 IsaacSim 场景组合契约（上游 PR [unilabsim/unisim#79](https://github.com/unilabsim/unisim/pull/79)）；合并后切换回 PyPI 发行版。

`third_party/simtoolreal_rl_games` 下的 vendored RL-Games 运行时经 SHA 门禁；不要改动其文件。其 pristine 源对象由 `vendor-pristine` 分支承载，保证审计在任何 clone 中可用——clone 时请 fetch 全部分支。push 前运行 `make check` / `make test`（428 项测试）。

## 致谢与支持

- **SimToolReal**——上游任务、资产与训练语义（[tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal)，MIT）。上游论文报告的数字（如 sim2real 性能）属于原实现，此处未重新验证。
- **UniLab** / **UniSim** / **unilab-rl**——训练基础设施、物理后端契约与异步运行时。
- **RL-Games**——SAPG 学习器，自上游 SimToolReal fork vendored 并带审计补丁集（`third_party/simtoolreal_rl_games/PATCHES.md`）。

许可：Apache-2.0（本仓库）；vendored 运行时与上游资产保留各自许可。欢迎 issue 与 PR——框架层问题（后端、契约）请报到 UniLab/UniSim。
