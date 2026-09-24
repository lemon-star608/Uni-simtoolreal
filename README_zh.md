# simtoolreal_unilab

[English](README.md)

迁移到 [UniLab](https://github.com/Motphys/UniLab) + [UniSim](https://github.com/unilabsim/unisim) 的 SimToolReal teacher 策略训练：KUKA iiwa14 机械臂（7 自由度）搭载 SHARPA 灵巧手（22 自由度），从 1,200 个程序化生成的工具变体池中学习抓取与举升工具，训练语义与[原仓库](https://github.com/tylerlum/simtoolreal)对齐。两个物理后端——IsaacSim（mapped 子进程 worker）与 MuJoCo（mjbatch VariantPack）——跑同一份 MJCF 资产（robot/table/floor 各一份、每个工具一份，一次生成）；学习器是 vendored、SHA 锁定的 RL-Games SAPG 运行时。

| 能力 | 状态 |
| --- | --- |
| `SimToolReal` teacher 训练（RL-Games SAPG） | 双后端全规模运行；任务/学习器语义与原仓库做过标签对拍 |
| 物理后端 | MuJoCo（默认）与 IsaacSim，同一份资产/任务定义，`--backend` 切换 |
| 与原 IsaacLab 实现的标签一致性 | 同 seed 12,288 env run 的关键 TensorBoard 标签族不可区分 |
| 回放 / 检查 | `simtoolreal-play`、与原仓库标签集一致的 TensorBoard 叠加 |
| student / 相机 / 传感器策略、部署、sim2real | 未迁移；本仓库不含部署代码 |

## 首次运行

要求：Linux x86_64、带 CUDA 的 NVIDIA GPU（学习器与 IsaacSim worker 共用）、[uv](https://docs.astral.sh/uv/)、约 20 GB 磁盘（worker 运行时与缓存）。依赖全部由 lockfile 解析（unisim-core 与 mjbatch-uni 走 PyPI；unilab 为上游 git 修订 pin）——无需额外克隆。

```bash
git clone <本仓库> && cd simtoolreal_unilab
uv sync
uv run simtoolreal-assets    # 校验 vendored 资产
```

`uv sync` 装不了的一件东西是 IsaacSim worker 运行时（NVIDIA SDK 不再分发）。运行一次 `bash scripts/setup_isaacsim_env.sh`（vendor 自 UniLab 官方安装器，IsaacSim 5.1.0 + IsaacLab 2.3.0，幂等）；`simtoolreal-train`/`simtoolreal-play` 会自动发现。想换别的 IsaacSim 解释器：`export UNISIM_ISAACSIM_PYTHON=/path/to/venv/bin/python`。

机器人/桌子的源 URDF 随仓库发布在 `assets/simtoolreal/`（上游 MIT，SHA-256 清单），首次使用时一次性序列化为实体 MJCF；1,200 个工具 MJCF 程序化生成并缓存在 `/tmp`（可用 `SIMTOOLREAL_TOOL_POOL_ROOT` 覆盖）。

## 训练与评估

```bash
# 正式训练（MuJoCo 默认，纯 CPU；两个后端默认同为 24,576 env = 6 x 4,096
# 探索块，跑 1,200 工具池）
uv run simtoolreal-train

# IsaacSim 后端（同一份任务定义；需要 worker 运行时与 GPU）
uv run simtoolreal-train --backend isaacsim

# 验收冒烟（64 env、24 工具池、2 epoch，几分钟；绝不作为策略基线）
uv run simtoolreal-train \
  algo.num_envs=64 env.tool_pool_num_per_type=2 \
  rl_games.params.config.max_epochs=2 \
  rl_games.params.config.expl_coef_block_size=32 \
  rl_games.params.config.minibatch_size=256 \
  rl_games.params.config.central_value_config.minibatch_size=256 \
  training.smoke=true training.no_play=true

# 回放 checkpoint（默认走 MuJoCo 回放；--backend isaacsim 经 IsaacSim worker 渲染）
uv run simtoolreal-play --checkpoint-file logs/rlgames_sapg/SimToolReal/<run>/nn/<checkpoint>.pth

uv run simtoolreal-list-envs                 # SimToolReal  mujoco / SimToolReal  isaacsim
uv run tensorboard --logdir logs/rlgames_sapg
```

产物落在 `logs/rlgames_sapg/SimToolReal/<run>/`（checkpoint、TensorBoard 事件、run 配置）。MuJoCo 线不需要 IsaacSim worker 运行时。


## 想改什么

| 改动目标 | 入口文件 |
| --- | --- |
| 奖励项与权重 | `src/simtoolreal_unilab/tasks/simtoolreal/rewards.py`、`conf/rlgames_sapg/task/simtoolreal/base.yaml` |
| 观测 / 目标指令 / 动作 | `tasks/simtoolreal/observations.py`、`commands.py`、`actions.py` |
| 复位、wrench DR、终止 | `tasks/simtoolreal/events.py`、`terminations.py` |
| 工具池与 MJCF 生成 | `tasks/simtoolreal/tool_catalog.py`、`tool_mjcf_writer.py`、`entity_mjcf_writer.py` |
| 实体 bootstrap 与变体绑定 | `tasks/simtoolreal/scene.py`、`env_cfg.py` |
| 算法超参 | `conf/rlgames_sapg/config.yaml` |
| 叠在官方 pin 上的行为 | `src/simtoolreal_unilab/_compat/`（上游长出即移除） |

依赖 pin：`unilab` 跟踪上游 main @ `838cc03e`（git rev，含 [#1620](https://github.com/Motphys/UniLab/pull/1620)/[#1636](https://github.com/Motphys/UniLab/pull/1636)；发版含它们后切回 PyPI）、`unisim-core >= 1.7.5`、`mjbatch-uni >= 0.2.4`。`third_party/` 下的 vendored RL-Games 运行时 SHA 锁定，禁止编辑。推送前跑 `make check` / `make test`。

## 致谢

- **SimToolReal**——上游任务、资产与训练语义（[tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal)，MIT）；论文报告的数字（如 sim2real 表现）属于原实现，此处未重新验证。
- **UniLab** / **UniSim** / **unilab-rl**——训练基础设施、物理后端契约与运行时；**RL-Games**——SAPG 学习器，带审计补丁集 vendored（`third_party/simtoolreal_rl_games/PATCHES.md`）。

许可证 [Apache-2.0](LICENSE)；vendored 运行时与上游资产保留其各自许可证。框架层问题（后端、契约）请报给 UniLab/UniSim。
