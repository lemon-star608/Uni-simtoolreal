# SimToolReal manager-based 迁移清单

> Step 0.3 只读探测结果。2026-09-11 建立。本文是任务迁移的证据清单，不表示
> 下列代码已经迁移或 IsaacSim worker 已经通过运行时探针。
> 2026-09-22 起第 7 节是当前资产/后端构造口径；前文保留当时的迁移证据与
> 历史 GoalViz 探针记录，不作为当前场景声明。
> 2026-09-23：URDF 运行时线已退役（URDF 冻结线见 tag `urdf-anchor`）；前文
> 以现在时出现的 URDF 描述（工具 URDF pool、URDF→USD 转换、fixed_model_variants
> 整模型通道等）均为历史记录。

## 0. 范围、来源和判定方法

本清单以 IsaacSim 原仓库 `simtoolreal/isaacsimenvs/tasks/simtoolreal/` 为任务
语义主源，以旧版 `simtoolreal-unilab-old/UniLab/.../simtoolreal/` 为 MuJoCo
旁证，并审计当前 `unisim/src/unisim/` 和 UniLab manager-based 先例。

快照：原仓库 `313d5aea`，UniLab `db1a6e5b`，UniSim `1ef3bb6`，旧版 UniLab
`5f4a16c2`。行号均指上述快照；源文件增加内容后应重新生成或修订本表。

覆盖标记：

* **有**：当前公开 `SimBackend` 接口和目标 adapter 已有可消费的路径。
* **部分**：接口存在但 IsaacSim adapter、数据形状、物化或运行时语义尚未完成。
* **缺**：没有任务可依赖的公开契约，或当前 IsaacSim 路径明确没有实现。
* **任务侧**：不应扩展 backend 契约；应落在 manager term、owner YAML 或冷路径
  资产物化中。

本轮没有启动 IsaacSim、没有生成 USD、没有修改 UniSim/MuJoCoUni，也没有把“代码
审计可行”写成“运行时验证通过”。

## 1. 映射表：原组件 → manager term 与 UniLab 落点

目标目录形态遵循现有 manager-based 任务和 `wuji_unilab` 先例：

```text
src/unilab/tasks/manipulation/simtoolreal/
  __init__.py scene.py actions.py commands.py observations.py rewards.py
  events.py terminations.py config.py tool_catalog.py DESIGN.md
src/unilab/conf/rlgames_sapg/task/simtoolreal/{base,isaacsim}.yaml
```

| 原实现组件（证据） | 语义 | manager term 类型 | UniLab 落点 | 备注 |
|---|---|---|---|---|
| `simtoolreal_env.py:31-81` | DirectRLEnv hooks、decimation、step 顺序 | 工厂 + 各 manager | `__init__.py`、owner YAML | 不建 SimToolReal env 子类；`make_manager_based_rl_env` 组装 |
| `scene_utils.py:1658-1821` | robot/table/object/goalviz/ground/light 冷启动 | scene/config | `scene.py`、`config.py`、IsaacSim owner | 资产转换和 USD bake 属 adapter/冷路径；goalviz 仅 IsaacSim |
| `action_utils.py:18-75` | canonical→Lab 关节排列、随机 delay、arm delta、hand absolute、MA、clamp | `ActionTerm` 自定义 `process_actions` | `actions.py` | 当前 IsaacSim utility 未在此处 clip `[-1,1]`；若依赖 rlgames `env.clip_actions=1.0` 必须在 0.1/Step 1 明确记录 |
| `simtoolreal_env.py:60-64` | pre-physics action + wrench 顺序 | action term + event/DR term | `actions.py`、`events.py` | wrench 需要公开 `SimBackend.apply_body_force` 或等价能力 |
| `action_utils.py:77-130` | object force/torque 采样、衰减、lift gate、写入 | interval/event term | `events.py` | 当前 `IntervalRandomizationPlan` 是通用 push/body force/torque 形态；须对齐质量与时序 |
| `obs_utils.py:136-150` | canonical joint pos/vel/previous targets | observation terms | `observations.py` | 关节名和排列在 init 冷路径解析并缓存 |
| `obs_utils.py:153-205` | palm/fingertip/object/goal 状态、keypoints、距离和 success 前置缓存 | observation + command/termination 共享 hub | `observations.py`、`commands.py`、`terminations.py` | 只能经 Entity/SimBackend facade；不能读 backend 私有对象 |
| `obs_utils.py:231-348` | actor/critic 字段、object state delay/noise、velocity noise、clamp | observation group/term | `observations.py` + `base.yaml` | actor 140、critic 162（按字段宽度计算）；clean/noisy 分流必须保持 |
| `obs_utils.py:352-468` | 可选 student camera/proprio | observation term（可选） | `observations.py`、`scene.py` | 非训练 teacher 默认路径；当前 0.3 只登记需求，不阻塞最小任务 |
| `reward_utils.py:8-23` | lift progress、一次性 lifting bonus、lift latch | reward term | `rewards.py` | `z_lift=0.05+z-z_init`；lift 后停止 progress reward |
| `reward_utils.py:26-51` | fingertip 距离 delta、keypoint 距离 delta | reward term | `rewards.py` | 分别由 `~lifted` 与 `lifted` gate |
| `reward_utils.py:54-64` | arm/hand joint velocity L1 penalty | reward term | `rewards.py` | 原字段名为 actions penalty，实际读取 joint velocity |
| `reward_utils.py:79-153` | reach bonus、reward tracker 更新与合计 | reward terms | `rewards.py` | bonus 可按 near-goal 摊销或 success 一次性发放 |
| `goal_sampling.py:21-62`、`reset_utils.py:304-369` | absolute/delta goal pose、四元数 wxyz | `CommandTerm` | `commands.py` | CommandTerm 是目标推进、成功计数和 reset 生命周期 hub |
| `reset_utils.py:200-220` | robot DOF reset、位置/速度噪声、previous target seed | reset event | `events.py` | 通过 Entity 写状态；不在 term 中扫描资产 |
| `reset_utils.py:223-301` | table/object pose、table z/XY/yaw、object velocity | reset event | `events.py` | object root pose/velocity 写入需要公开 facade |
| `reset_utils.py:372-401` | goal trackers、delay queues、wrench buffers、success/lift trackers | command/recorder/事件状态 | 各 term 实例 | 跨 term 用 `env.<manager>.get_term(...)` 回取，不设 env 全局状态 hub |
| `termination_utils.py:10-36` | tolerance curriculum | 状态在 command hub，逐步触发在 termination term | `commands.py`（状态）+ `terminations.py`（触发） | 每 interval 按上一 episode successes 平均值缩小 tolerance；tick 先于同帧 success 判定 |
| `termination_utils.py:39-72` | success 后换目标、fall、hand far、max successes、timeout | termination terms | `terminations.py` | success 不直接结束 episode；成功后清 goal tracker 并置零 episode length |
| `logging_utils.py:1-33` | reward/termination metrics | recorder term | `events.py` 或 `RecorderTerm` | 不在 backend 里实现任务日志 |
| `simtoolreal_env_cfg.py:538-595`、`cfg/task/SimToolReal.yaml` | cfg 分组与默认值 | Hydra config | `base.yaml`、backend leaf | task 语义和 owner physics 在 base；sim/backend 旋钮在 leaf |
| `utils/generate_objects.py:1-406`、`object_size_distributions.py:1-234` | 12 分布、seed=42、工具 URDF、质量/惯量 | 后端中立 catalog + adapter serializer | `tool_catalog.py`；URDF writer 属 step 1 | 0.3 只登记接口；不在此步实现 writer |
| `simtoolreal/isaacsimenvs/cfg/train/SimToolRealSAPG.yaml` | PPO/SAPG、asymmetric critic、normalization、bootstrap | algo config | `src/unilab/conf/rlgames_sapg/` | SAPG 基建在 0.1；这里作为语义对拍基准 |

### 生命周期顺序（需要在实现和对拍中固定）

原 Isaac Lab `DirectRLEnv` 的实际顺序是：处理 action → decimation physics →
更新状态 → `_get_dones()` → `_get_rewards()` → 对 done env 执行 `_reset_idx()` →
`_get_observations()`。因此 SimToolReal 的成功检测在 reward 之前，成功 bonus 仍
能在成功帧发放；目标位姿的正式写入必须在 reward 读取成功状态之后完成。

当前 UniLab `ManagerBasedRlEnv` 的顺序是：处理 action/physics → termination manager
→ reward manager → step/interval events → command manager → observation manager；
done env 随 auto-reset 尾部（或显式 `reset()`）进入 ManagerBasedRlEnv 的 reset
transaction。两者的公共骨架不需要重排。实现时
应让 termination term 只记录本步的 `success`/`goal_advance_pending`，reward terms
消费这个快照，CommandTerm 在 reward 之后推进 goal、清 per-goal tracker；不能在
termination term 内提前写 GoalViz 或清掉 reward 所需状态。

这保证了“成功帧奖励”和“下一目标从下一次读取开始生效”与原 DirectRLEnv 等价，
同时保留 UniLab 通用类的 manager 顺序。MuJoCo 的 engine-level `was_autoreset` 是
另一个输入，不能冒充 episode reset；IsaacSim 对应项为空。

**框架改动判断**：不要为了 SimToolReal 修改 `ManagerBasedRlEnv` 的全局顺序，也
不要让 termination term 直接调用 reset。需要的只是 SimToolReal 自己的 term 状态
协议：termination 写入 pending success，reward 读取快照，CommandTerm 在其既有
post-compute 阶段推进目标，reset manager 处理真正 done 的 env。只有在最小探针证明
现有 `CommandManager` 无法提供“reward 后、observation 前”的提交点时，才考虑给
manager 增加一个通用的 post-reward hook；这个 hook 必须对所有任务保持 no-op 兼容，
不能为 SimToolReal 硬编码。

## 2. 分叉点清单：应落在哪里

> 注（2026-09-23）：表中「IsaacSim 复用原仓库 URDF→USD」「IsaacSim=URDF」
> 「1200-tool pool 经 `fixed_model_variants` 整模型通道」等现在时 URDF 描述
> 已退役（URDF 冻结线见 tag `urdf-anchor`）；当前口径见第 7 节。

| 分叉/差异 | owner/adapter 归属 | UniLab 任务是否分支 | 当前结论 |
|---|---|---|---|
| 后端选择与 device/solver/worker 参数 | `conf/.../simtoolreal/isaacsim.yaml` + adapter | 否 | base.yaml 不写引擎判断 |
| robot/table/goalviz 资产 | owner scene cfg + IsaacSim adapter | 否 | IsaacSim 复用原仓库 URDF→USD；MuJoCo 未来用 MJCF |
| 工具变体文件格式 | `FixedModelVariantCfg`（`name` + `source_model_file`，`unilab.base.variants`），工厂 guard fail-closed | 否 | IsaacSim=URDF，MuJoCo=MJCF；SimToolReal owner 已接入 1200-tool pool |
| 1200 个变体及 round-robin 分配 | `tool_catalog` + `EnvCfg.fixed_model_variants`（catalog）→ 上游 guard 写 `SceneCfg.fixed_variant_plan` | 否 | owner 产确定性 catalog，guard 生成 round-robin assignment，worker 在 INIT 物化并固定 |
| 多资产角色（robot/table/object/goalviz） | `SceneCfg.entities` / adapter scene assembly | 否 | `SceneCfg` 管“有什么”，variant 管“每 env 有何不同” |
| goalviz | IsaacSim owner/adapter | 否 | MuJoCo 暂不做，避免未验证变量 |
| URDF collision marker、fixed-joint、self-collision FilteredPairs | IsaacSim converter/bake | 否 | 格式无关的物理补偿留在 adapter 冷路径 |
| robot PD/armature/frictionloss | owner actuator config + adapter runtime write | 否 | manager 只提交目标；不访问 `ImplicitActuator` 私有实现 |
| friction / material bucket | adapter runtime PhysX view | 否 | 配置是任务参数，写入是 adapter；当前原仓库对象/桌/robot=.5、指尖=1.5 |
| solver、contact/rest offset、max depenetration | owner sim cfg + adapter bake | 否 | MuJoCo `<option>` 不映射到 PhysX 数值 |
| observation/action/object delay、wrench DR | term 实例/DR provider | 否 | 公开 contract 只承载后端物理写入；随机采样属于任务语义 |
| NaN engine autoreset | MuJoCoUni + MuJoCo adapter | 否 | IsaacSim 无对应物；信号转 termination 后才走正式 reset |
| reset 时机与 final observation | UniLab lifecycle + owner adapter policy | 否 | IsaacSim 对拍原仓库丢弃终端观测；MuJoCo 对拍旧实现保留 `final_observation` |
| camera/student observation | task optional term + IsaacSim scene | 否 | 不进入 0.3 最小 contract gate |

原则：owner YAML 只声明后端身份、旋钮和能力禁用；任何“如果是 mujoco/isaacsim”
的任务公式分支都视为边界泄漏，应改成公开契约能力或 owner term enable/disable。

## 3. 接口需求与契约覆盖清单

### 3.1 冷路径、场景和资产

> 注（2026-09-23）：本表「owner 生成 1200 个 URDF」「四角色 URDF/RigidObject
> 组装」「URDF 初始 pose」等现在时 URDF 描述已退役（URDF 冻结线见 tag
> `urdf-anchor`）；当前口径见第 7 节。

| 原需求 | 当前公开契约 | 覆盖 | IsaacSim 起点/缺口 |
|---|---|---|---|
| 场景源、实体角色、fragment/keyframe | `SceneCfg.model_file`, `entities`, `fragment_files`, `default_keyframe_name`（`unisim/scene.py:41-55`） | 部分 | IsaacSim worker 目前按单 MJCF/单 articulation 设计；需支持 URDF 场景多 root 角色 |
| 变体集合与每 env assignment | `EnvCfg.fixed_model_variants`（`FixedModelVariantCatalogCfg`）→ guard 产 `SceneCfg.fixed_variant_plan`（`FixedVariantPlan.variants`/`assignment`，int32 round-robin） | 有 | owner 生成 1200 个 URDF 与 catalog；worker INIT 校验后物化 |
| robot/table/object/goalviz 独立物化 | `SceneCfg.entity_assets` + `SceneCfg.entities` typed cold-path contract | 有 | worker 四角色 URDF/RigidObject 组装、root/readback 已由 D2 Kit 探针验收 |
| named keyframe/default qpos | `get_keyframe_qpos`, `get_default_qpos/dof_pos`（`backend/base.py:689-713`） | 部分 | IsaacSim 需把 URDF/owner 初始 pose 映射成统一状态；任务当前不依赖 MuJoCo keyframe |
| body/link name resolution | `get_body_ids`, `get_body_id`（`base.py:737-753`） | 部分 | adapter 必须冷路径建立 robot palm/fingertip/body id，不能热路径 XML 查找 |
| joint names/ranges/actuator targets | `get_joint_range`, `get_actuator_*`（`base.py:645-667,858-865`） | 部分 | IsaacSim actuator metadata 与 29 DOF canonical mapping 尚未完成/实测 |
| friction/material、mass、armature 等 cold metadata | getter/DR payload 有 MuJoCo 形态（`base.py:820-838`; `dr/types.py:157-210`） | 部分 | IsaacSim friction/armature 已由 Kit 探针读回；1200 variant mass 由 worker 实测、经 `get_entity_variant_metadata` 读回进入 wrench term，reset DR 仍不纳入 v1 |

### 3.2 热路径状态读取、控制和 reset

| 原需求 | 当前公开契约 | 覆盖 | 说明 |
|---|---|---|---|
| policy step + decimation | `SimBackend.step(ctrl, nsteps)`（`base.py:871-882`） | 有 | host/worker 需证明 nsteps 与 2×1/120 时序一致 |
| qpos/qvel snapshot | `get_state` 拼接 base pose/quaternion/DOF（`base.py:573-600`） | 有 | 公共表示是 wxyz；IsaacSim worker 需转换并验证形状 |
| selected env reset / set state | `reset`, `set_state`（`base.py:602-615,905-932`） | 有 | object/table/goalviz root pose/velocity transaction 已接通；D2 验证 goalviz-only reset 隔离 |
| base/object pose、linear/angular velocity | `get_body_*_w`, `get_body_state_w` 及 rows（`base.py:1220-1310`） | 有 | robot body 与独立 RigidObject root 都已映射；D2 验证 object/goalviz per-env 读回 |
| body pose/velocity used by fingertips/palm | 同上 | 部分 | 需明确 link state 的 world-frame angular velocity及 array shape；IsaacSim 需实测 |
| write robot joint targets | `step(ctrl)` direct control + `set_pre_step_control`（`base.py:884-900`） | 部分 | position-target/PD semantics 是 owner+adapter 组合；`apply_action_pipeline` 不能直接调用 Isaac Lab object |
| external body force/torque | `apply_body_force`（`base.py:981-1002`）和 interval plan | 有 | host/worker dense wrench slots 及 `(N,1,3)` 世界系效果已由 D2 Kit 探针验收 |
| reset pose/velocity writes | `set_state` 只覆盖 generalized qpos/qvel | 有 | scene entity facade 已把独立 rigid root 写入同一 reset transaction |
| sensor binding | `BackendSensorView`, `bind_sensor_data`（`base.py:397-475,1503-1564`） | 部分 | SimToolReal teacher 主要 state，不需要 contact sensor；student camera 是可选 IsaacSim term |
| NaN/autoreset mask | 当前 contract 无通用引擎 autoreset getter | 缺（IsaacSim不需要） | MuJoCo 专属能力应由 adapter/termination 输入暴露，不能放公共 episode reset |

### 3.3 DR、物理属性和渲染

| 原需求 | 当前契约 | 覆盖 | 结论 |
|---|---|---|---|
| reset DR terms | `ResetRandomizationPayload` 覆盖质量/重心/惯量/摩擦/KP/KD/geom 参数（`dr/types.py:157-210`） | 部分 | 当前 SimToolReal v1 的 reset pose/target/noise 仍是任务 term；PhysX 有效写入需 adapter 补齐 |
| interval force/torque | `IntervalRandomizationPlan` + `apply_interval_randomization`（`dr/types.py:217-270`; `base.py:947-1002`） | 有 | dense slots、finite 校验、Kit 效果和按 1200 变体真实质量缩放均已验证 |
| geometry size/solref/solimp | getter + reset payload | 部分 | IsaacSim URDF→USD 不同字段不直接存在；contact/rest offset 走 bake；不能伪造 MuJoCo solref |
| runtime friction | `get_geom_friction` 主要是读接口 | 有（IsaacSim写入） | IsaacSim worker 通过 PhysX view 写入并读回；B2/C2 覆盖 robot/table/object |
| bake kinematic/gravity/collision/solver/filter | 无通用 SimBackend 方法 | 有（IsaacSim adapter） | worker 冷路径完成 USD bake、FilteredPairs 与 role-specific PhysX 属性；不扩张 manager term |
| goalviz kinematic no collision | 无通用 semantic method | 有（task+adapter） | IsaacSim spawn 第二个 RigidObject 并在 reset 写 pose；D2 验证其运动学稳定；MuJoCo owner 关闭 |
| playback/debug overlays | `BackendPlayCapabilities`, `DebugPrimitive`（`base.py:41-153,478-500`） | 有（接口） | 与训练最小路径无关；不要因此引入 goalviz 通用抽象 |

### 3.4 当前 IsaacSim adapter 的基线（避免把“接口有”误认为“adapter 有”）

当前 `unisim/backend/isaacsim/worker.py` 已能从 typed `entity_assets` 物化
Articulation(robot) + RigidObject(table/object/goalviz)，并保留 `MultiUsdFileCfg`
变体入口；`subprocess_ipc/backend.py` 也按实体声明分配 root/reset/wrench slots。
这些路径已经由 D2 Kit 探针完成运行时验收，E1 又验证了启用 `wrench_dr` 后
ManagerBasedRLEnv 的 reset、interval event 和 10 步控制闭环。

已有的单 robot 能力是：position actuator 名称/range/gains 解析与
`ImplicitActuatorCfg` position target（`subprocess_ipc/backend.py:835-884`、
`isaacsim/worker.py:287-303,447-469,554-571`）；robot root/link world pose、quat、
linear/angular velocity 与 masked robot root/joint reset（`isaacsim/worker.py:
524-624`）。以下能力目前明确 fail-closed 或未实现：

* `set_state` 仍拒绝非空 `ResetRandomizationPayload`（`isaacsim/backend.py`、
  `subprocess_ipc/backend.py`）；root pose transaction 与 interval wrench 是独立
  已实现通道，不能把 reset DR dataclass 的存在算成 PhysX 属性随机化支持。
* wrench host/worker dense slot、每子步外力写入和 control step pending 清除已通过 D2
  Kit 探针；E1 已启用 owner YAML 的 `wrench_dr` 并跑通 ManagerBasedRLEnv。真实工具
  质量已由 1200-tool init plan 传入并在 term 内按 env 缩放。
* contact-force sensor 在 worker 中是零槽位，host 显式将 contact sensor 标成
  unsupported；这不满足原任务真实接触力需求（当前 SimToolReal teacher 路径不
  依赖 contact sensor，但清单必须保留该事实）。
* worker 已实现 URDF converter、USD bake、FilteredPairs、自碰撞相邻 link 过滤、
  role-specific gravity/kinematic、solver/contact/rest offset 和 PhysX material
  runtime write；这些能力由 B2/C2/D2 探针覆盖。原仓库对应实现位于
  `scene_utils.py:1309-1539,1546-1624`。
* 当前 IPC 已传独立 rigid-object root/reset/wrench slots；仍没有 depth camera
  contract 或 engine-exception autoreset/restart。
* 原仓库机器人转换使用 `fix_base=True`（`scene_utils.py:1721-1725`）。当前
  IsaacSim worker 已支持 fixed-base robot articulation 与 floating object roots；
  rigid root layout 由实体扩展槽提供，D2 已验证 object/goalviz 的根状态读写。

因此第 4 节同时区分三种动作：需要先扩展 **契约** 的项目（typed asset source、
per-entity root/rigid-object state、wrench/reset property transaction）；只需补
**IsaacSim adapter/worker** 的项目（single-robot 已有路径之外的多资产、bake、
material、solver、filtered pairs）；以及必须留在 **任务/owner** 的项目（goal、
reward、delay、success、episode lifecycle）。

### 3.5 三方对拍时必须显式登记的已知偏差

* 原 IsaacSim utility 的当前 action 路径在 `action_utils.py:31-34` 只做排列，旧
  MuJoCo port 在进入排列/延迟前还会 clip `[-1,1]`；这不是可以默默选择的实现细节，
  应由 Step 1 对拍决定并写入偏差表。
* 原始 gym 代码曾计算但以零系数关闭 hand delta/object linear-angular velocity
  penalty；当前 IsaacSim utility 明确只保留 7 个 reward 项。不要从 gym 历史字段
  自动恢复已关闭项。
* 当前 IsaacSim 配置同时有 `fixed_size_keypoint_reward` 和按对象尺度的观测
  keypoints；success/reward 路径必须继续使用 fixed-size 分支（默认开启），观测
  路径才使用每 env scale/noise。
* 原 IsaacSim DirectRLEnv 同帧 reset 后返回 reset 观测；旧 MuJoCo manager port
  保留 `final_observation`。这属于 owner/lifecycle 口径，不能用一个统一 autoreset
  名称掩盖。

## 4. IsaacSim 缺口 checklist（从第 3 节“缺/部分”抽出）

> 注（2026-09-23）：本节的 URDF converter/URDF 解析相关验收项描述的是已退役
> 的 URDF 运行时线（URDF 冻结线见 tag `urdf-anchor`），保留为历史验收证据。

以下是 Step 1 的可执行验收项。状态“待实测”不是“失败”，但在通过前不能宣称
IsaacSim 迁移完成。

### 契约/host/worker

- [x] 变体通道为 `SceneCfg.fixed_variant_plan`（`FixedVariantPlan`：factory guard
  从 task 声明的 `fixed_model_variants` catalog 构建），变体携带 `model_file`
  源文件，校验变体数量与 assignment 范围；其余格式 fail-closed。
- [x] `IsaacSimBackend` 消费 `SceneCfg.entity_assets` 与 `fixed_variant_plan`：
  `consumes_fixed_variant_pool` 在冷路径对称绑定 plan 与实体角色，变体质量经
  `get_entity_variant_metadata` 读回；不把 task term 变成 SDK 对象。
- [x] worker 支持同一 env 同时 materialize Articulation(robot) 和 RigidObject
  (table/object/goalviz)，并确保 `replicate_physics=False`、`clone_in_fabric=False`
  时每个 env 都有实体。
- [x] worker 按 `model_assignments` 实现静态 round-robin；reset 不重编译、不解析
  URDF、不改变 variant assignment。
- [x] host/worker 通过共享内存传递 qpos/qvel、root entity states、control 和
  selected env ids，明确 batch-first shape 与 wxyz/world angular velocity。
- [x] backend `set_state`/worker state slots 对 qpos/qvel 和读取状态做 finite 校验；
  worker/host 在边界校验 ID、shape 和 finite，不能把上层 NanGuard 当成物理状态契约。
- [x] adapter 在 materialization 缓存 palm/fingertip/body/joint/actuator 映射；热
  路径没有 XML/SDK name lookup。

### 资产转换与物理有效状态

- [x] URDF converter 按原仓库 flags 调用：`force_usd_conversion=True`、
  `fix_base`、`merge_fixed_joints=True`、`replace_cylinders_with_capsules`、
  `joint_drive`、robot `self_collision=True`（证据 `scene_utils.py:1421-1449`）。
- [x] URDF SDF collision markers、相邻 link `FilteredPairsAPI`、mesh collision
  形状和碰撞参数写入 bake；证据 `scene_utils.py:1172-1365`。
- [x] bake 写入 kinematic、disable gravity、max depenetration velocity、
  articulation/self-collision、solver iterations、contact/rest offset、
  collisionEnabled；证据 `scene_utils.py:1461-1539`。
- [x] robot 的 PD stiffness/damping/armature/frictionloss 通过公开 adapter
  通道生效；不能只验证 cfg 被构造。必须用 PhysX view/pxr 读回有效值。
- [x] robot/table/object/goalviz friction runtime 写入与配置一致；覆盖原仓库
  robot/table/object `[0.5,0.5,0]`、指尖 `[1.5,1.5,0]` 及 bucket DR；证据
  `scene_utils.py:1546-1624`。
- [ ] 资产探针 A 对照同一 URDF、同一 `UrdfConverterCfg` 的 USD 属性（mass、CoM、
  inertia、collision、joint limits）；探针 B 读回 bake/摩擦/solver/drive 有效值。

### 任务接口与生命周期

- [ ] robot joint target 写入支持 29 DOF canonical policy order，保留 action
  delay、arm delta、hand absolute、moving average、joint limits 和 decimation。
- [x] object global force/torque 写入支持每 env `(N,1,3)`，与前一帧 lift latch、
  log-uniform probability、decay 和 selected env reset 一致。
- [x] object/table/goalviz root pose/velocity reset 是同一事务语义；goalviz reset
  不改变 object 或 generalized robot state。
- [x] 读取 robot palm/fingertips、object root、goalviz root 的 world pose/quaternion/
  velocity，支持 keypoint 计算所需批量 shape；四元数固定 wxyz。
- [ ] `fall | max_successes | hand_far` 与 timeout 的 terminated/truncated 分离；
  timeout 按 SAPG `value_bootstrap=True`，真正 failure 不 bootstrap。
- [ ] success 帧由 termination 记录并由 reward 消费；reward 后由 CommandTerm 同帧
  换 delta goal、清 per-goal trackers、将 episode length 置零；success 本身不作为
  termination 原因。
- [ ] student camera/goalviz 隐藏、render/debug 只作为可选能力，不能阻塞 teacher
  state-only 最小场景。

## 5. 0.3 完成判据与后续输入

0.3 的文档产物已经具备四个视图，但“完成迁移”还需要：

1. 0.2 的 worker 最小场景结果补入第 4 节对应复选框；
2. 0.4 探针 A/B/C 结果补入本文件和主 discussion 的议题 F；
3. Step 1 实现时逐项把“部分/缺”改为证据链接，而不是凭构造成功推断物理生效；
4. 任务代码审查确认所有后端差异都停留在 owner YAML、UniSim adapter 或冷路径
   资产物化，未出现 backend-private feature leakage。

## 6. 后续讨论结论（2026-09-11）

本节记录 0.3 清单完成后的两轮设计讨论。它们是对第 3、4 节的解释和实施边界，
不是“已实现”声明。

### 6.1 场景、实体和 fixed-base robot

（初始设计记录）IsaacSim worker 当时仍然是单个 MJCF → 单个 `Articulation`，而 SimToolReal 的
目标场景包含 robot、table、object、goalviz 四个逻辑实体。原仓库机器人转换使用
`fix_base=True`，当前 worker 却依赖 floating root/freejoint 的 root-state layout；
因此 G1 MJCF 最小 smoke 通过，不能证明 SimToolReal fixed-base URDF 链路可用。

实施方向：

* unisim 的冷路径场景契约表达资产角色和物化类型：fixed-base articulation、dynamic
  rigid object、kinematic rigid object；不要让 unisim 解析 UniLab 私有 `EntityCfg`。
* IsaacSim adapter 按角色物化 `robot`、`object`、`table`、`goalviz`，并为每个 env
  建立稳定的 entity/root/body 映射。robot 不要求 floating root；object 的 root
  状态独立于 robot generalized state。
* `SceneCfg.entities` 继续表达“场景里有什么”，变体计划表达“每个 env 哪里不一样”。
  不把工具池塞进 `SceneCfg`。

### 6.2 工具变体与分配

> 注（2026-09-23）：本小节描述的 `fixed_model_variants` 整模型通道与工具
> URDF pool 已退役（URDF 冻结线见 tag `urdf-anchor`）；当前 B-form 实体
> 变体通道见第 7 节。

变体通道已落地：task 在 `fixed_model_variants` catalog 里声明完整工具文件池
（每变体一个 `source_model_file` URDF），factory guard 把它构建成
`SceneCfg.fixed_variant_plan`；实体侧用 `consumes_fixed_variant_pool` 声明被
替换的场景角色（IsaacSim 工具 URDF；MuJoCo 家族不经此通道）。

ToolSpec 采样、序列化和 `model_assignments` 由任务侧负责；adapter 只负责校验和
物化传入的 assignment。round-robin 是当前任务生成 assignment 的策略，不能在
adapter 内写死为唯一分配规则。变体在初始化时固定，reset 不重新编译或改变分配。

### 6.3 状态、reset 和 wrench

`SimBackend` 已有 `get_body_*`、`get_root_state_layout`、`set_state` 和
`apply_body_force(body_ids, force, torque)` 公共接口。因此这里的主要问题不是再
造一套任务专用 API，而是实现 IsaacSim adapter 的多实体映射、共享内存 slots 和
批量 shape。

* robot 的 29 DOF position target、palm/fingertip body state 和 fixed-base reset
  走现有 Entity/SimBackend facade。
* object/table/goalviz 需要独立 root pose/velocity 的批量读写；goalviz 是目标的
 物理可视化副本，不能改变 object 或 robot state。
* wrench 由任务 term 负责采样、衰减和 lifted gate，adapter 负责把世界系 force/
  torque 写入即将到来的 physics step，并在 reset 时清除 pending wrench。
* qpos/qvel、root/entity state 必须在 host/worker 两端明确 batch-first shape、
  wxyz quaternion 和角速度坐标系；状态写入和读回都应做 finite 校验。

### 6.4 物理属性和验收顺序

IsaacSim 资产物理属性按三层实现和验证：

1. URDF converter 使用原仓库相同 flags；
2. bake 阶段写入 kinematic、gravity、collision、FilteredPairs、contact/rest
   offset 和 solver 属性；
3. worker 启动后写入实际 PD/armature/material，并通过 PhysX view 或 pxr 读回。

探针 A 验证 URDF→USD 转换调用和属性，探针 B 验证运行时有效物理状态，探针 C
验证多资产 spawn 和变体分配。构造出 cfg 或 USD 文件不等于物理属性已经生效。
质量随机化、contact sensor、depth camera 和 worker 自动重启不属于当前 teacher
训练路径的必要前置，不应和这条最小链路捆绑实施。

### 6.5 任务生命周期与通用类边界

原 DirectRLEnv 的顺序是：

```text
physics → _get_dones() → _get_rewards() → reset done env → _get_observations()
```

当前 UniLab manager-based 环境的顺序是：

```text
physics → termination → reward → events → command → observation
```

不应为了 SimToolReal 重排 `ManagerBasedRlEnv` 的全局顺序，也不应让 termination
term 直接调用环境 reset。保持通用类不变时，SimToolReal 使用以下 term 内部协议：

* termination term 计算 success，写入本步 `pending_success`，暂不换 goal 或清除
  reward 所需 tracker；
* reward terms 读取该快照，完成成功帧 bonus 和其他 reward；
* reward 之后由 CommandTerm 推进下一目标、清理 per-goal tracker，并更新 hub 中
  的权威 goal；headless B-form 不提交 GoalViz entity 位姿，真正 done env 仍由
  通用 reset transaction 处理。

这样可以保持原任务的成功帧奖励和下一目标切换语义，同时不破坏其他任务依赖的
manager 生命周期。只有最小探针证明现有 CommandManager 没有 reward 后、observation
前的提交位置时，才考虑增加通用的 post-reward hook；该 hook 必须对其他任务默认
no-op，不能硬编码 SimToolReal 行为。

1.4b-1 探针结论（`manager_based_rl_env.py:498-539`，GoalViz mirror 仍在场景中时）：
提交点存在——
`command_manager.compute` 在 reward 之后、observation 之前执行，且包裹在
`_reset_state.scoped(...)` 里，§6.5 协议零框架改动落地。但写入通道有语义边界：
`ResetStateTransaction._prepare_state_write` 对 dirty env 行从**默认** qpos/qvel
铺底，`commit` 经 `SimBackend.set_state` 整行上传（reset_state.py:1305-1330,
914-945）。reset 路径上这是对的（reset event 把 robot/object 状态铺进同一行），
但 episode 中途的 success 换 goal 若经此通道提交 GoalViz，会把成功 env 的
robot/object 状态重置为默认值。因此 `SimToolRealCommand` 只在 reset 路径提交
GoalViz 位姿；episode 内的推进只更新 hub `goal` 缓冲，等 1.4s 场景接线时提供
partial-row 写入通道（如 mocap binding）。同一步 success 恰好到达 max episode
length 时，truncation 已在 termination 阶段算完，time-out term 需用
`pending_success` 掩码（1.4b-3 已落地，见本节末）以保持"成功步不截断"语义。

action clipping 继续由训练配置/rlgames wrapper 的 `clip_actions=1.0` 负责，任务
action term 不应在不登记语义变化的情况下重复裁剪。reward manager 则需确认
`scale_rewards_by_dt` 与原仓库的奖励尺度一致。

1.4b-2 结论（`reward_manager.py:119-150`）：`scale_rewards_by_dt: true` 时
`reward_buf = raw * weight * step_dt`，即 UniLab 每项奖励统一为原 DirectRLEnv
的 1/60（60 Hz）。逐 term 同比例缩放，在 advantage normalization 下策略不变；
但日志 episode return 与 critic obs 的 `reward` 字段（读 `env.reward_buf`）都比
原仓库小 60×。保留框架默认，不在任务侧反向补偿；已登记进 rewards.py 模块
docstring 偏差清单。同轮确认的另一处顺序语义：原仓库 `compute_rewards:97` 先写
`lifted` latch 再 gate fingertip/keypoint 项，manager 版各 term 改用幂等式
`(z_lift > threshold) | hub.lifted` 自行推导 gate，latch 只由 LiftingBonus 写，
因此 reward term 在 manager 配置里的排列顺序不影响数值。

1.4b-3 结论：termination 层五项已按上述协议落地（terminations.py）。
success 判定链（obs_utils.py:190-205 + reward_utils.py:79-84）由
`SuccessTracker` 在 termination 阶段计算并写 hub 快照（near_goal /
near_goal_steps / pending_success），success 不输出 done；time_out 用
`pending_success` 掩码替代原仓库"先置零 episode_length_buf 再判
truncation"（termination_utils.py:50-51, 65）的顺序依赖，成功帧不截断、
后续步由 command 阶段写入的零长度承接，逐帧语义与原仓库一致。
`MaxConsecutiveSuccesses` 比较 `goal_count + pending_success`：原仓库在
同一调用内先 `_successes += is_success`（:46）再比较（:58），而 UniLab 的
goal_count 在 command 阶段才递增，折入 pending 快照复现原"第 50 次成功
当步终止"的语义。`_termination_reasons` 字典（:66-71）不移植——
TerminationManager 的 per-term done 缓冲与 `Episode_Termination/` 日志已
覆盖。

热路径复查（2026-09-12）已按原仓库的
`compute_intermediate_values()` / `compute_rewards()` / `build_observations()`
调用链对照：SimToolReal command hub 在一个 control step 内缓存一次
fingertip 距离、keypoint 最大距离，并让 reward/termination terms 复用；policy 和
critic observation terms 也复用同一份 clean frame。缓存只存在于任务 term/hub，按
step 和 reset-row ID 失效，不改变通用 manager 生命周期、reset transaction、随机数
draw、字段顺序或数值公式。metrics/recorder 对本任务为空配置，运行时是
`NullMetricsManager`/`NullRecorderManager`，没有隐藏的逐步计算。原仓库的
interval wrench 位于 pre-physics hook，而通用 UniLab event 仍在现有 lifecycle 的
interval 阶段触发；这是已登记的时序差异，不能误报成额外 reset 或重复计算。
done 行仍会经历通用生命周期的“全量 step observation + reset-row observation”两次
组装；后者已经限制为 selected rows。要像原仓库一样只保留 reset 后的一次最终观测，
需要改通用 autoreset 顺序，不属于 task term 的无痛微调。

对齐审计修复（2026-09-28 差异清单及后续逐项复跑，均为 term 内部修复、不改
§6.5 相位顺序）：

* A1.2：原仓库在 `_get_dones` 内先清 per-goal tracker 再算奖励，success 帧的
  keypoint-delta 被 sentinel 填 0；manager 版 reward 先于 command advance，
  `KeypointDistanceDelta` 改为按 termination 阶段写入的 `pending_success`
  快照把成功行的 delta 置 0，tracker 的 min 更新保持不变。
* A1.5：同一 tracker-clear 顺序也作用于 fingertip-delta。原仓库
  `reset_goal_trackers()` 同时清 `_closest_fingertip_dist`，success 帧
  `distance_delta_reward()` 的每指 delta 都从 -1 sentinel 出发并 clamp 为 0；
  manager 版同样用 `pending_success` 掩码，不能用只修 keypoint 的局部结论覆盖。
* A1.6：成功换 goal 时只把 `episode_length_buf` 写 0 不够。`NpEnv.step()` 在
  read phase 返回后还会对 authoritative `state.info["steps"]` 加一；若 command
  阶段把它也写 0，下一帧读到 length=2，而 DirectRLEnv 在 post-step 加一后于
  `_get_dones()` 内清零、下一帧读到 length=1。goal advance 需把 authoritative
  计数器 staging 成“本帧后加一为 0”的值（有符号 -1 / 无符号 max），下一帧
  才恢复原仓库的 length=1。
* A1.3（2026-09-28 复查后撤销，不改门控行为）：初判认为 interval event 读
  当步闩锁会让力“举起当步就受力”，但该前提不成立——event 在本步物理与
  reward 之后触发，写入的 wrench 由 adapter staging 给即将到来的 physics
  step（`Entity.apply_body_wrench_to_sim`；backend 在下一步物理开始时消费，
  跨该步 substeps 后清除）。因此 event 写入覆盖的物理窗口与原仓库
  `_pre_physics_step` 写入覆盖的窗口相同，且读到的 `hub.lifted` 正是原仓库
  在该处读到的同一个 reward 阶段闩锁：直接读 `hub.lifted` 已是逐帧等价，
  自存一步延迟闩锁反而比原仓库晚一帧，已回退（保留 A1.2/A1.4）。
* A1.4：容差课程状态仍由 command hub 持有，逐步 tick 的触发移到
  `SuccessTracker` 内、先于同帧 success 比较（与原仓库
  `update_tolerance_curriculum` 在 `_get_dones` 内先行的顺序一致）；command
  阶段的 tick 已移除，`_last_curriculum_step` 守卫保证同帧重复调用不重复
  收紧。

### 6.6 当前实现快照（2026-09-12）

> 注（2026-09-23）：本快照中的 1,200-tool URDF pool 已退役（URDF 冻结线见
> tag `urdf-anchor`）；当前实现为 entity-split MJCF B form，见第 7 节。

该小节记录 2026-09-12 的 Step 1 快照：当时 `SimToolRealEnvCfg` 负责 seed=42、
12 类×100 的 1,200-tool URDF pool 与 manifest；manager factory 在 backend
materialize 前提交 `InitRandomizationPlan`；IsaacSim worker 在 round-robin
assignment 下只建立 1,200 个唯一 prototype，再复制到各环境。
`ModelVariantSpec.mass` 作为 owner metadata 随 INIT 传递，wrench term 使用
assignment 展开的 per-env mass。

真实 Kit 证据包括 D2 root/wrench 6/6、E1 Manager wrench、完整 1,200-tool pool 的
物化，以及 2048/12288-env 的 1 epoch smoke（均完成 checkpoint load）。正式
24576-env teacher training 仍是下一步；student camera、contact sensor、depth
camera 和 engine exception autoreset 保持为后续能力，不阻塞本 teacher gate。

## 7. 当前资产构造与后端路由（2026-09-22）

当前分支不再把 MJCF 视为 MuJoCo 专属路线，也不再在 IsaacSim 运行时调用 URDF
importer。两个已注册后端共享同一个 B-form 资产真值点：

1. `entity_mjcf_writer.py` 从 `build_robot_mjcf`、vendored table URDF 和工具
   catalog 写出 `robot.xml`、`table.xml`、`floor.xml` 与 1,200 个 standalone tool
   MJCF；PD/friction 等 owner physics 在源资产中只写一次。
2. `SimToolRealEnvCfg.bind_entity_variant` 在 factory 知道 `num_envs` 后，把工具
   catalog 绑定到 `object` entity，assignment 固定为
   `arange(num_envs) % variant_count`；质量/尺度表保留在任务侧并按 catalog 顺序
   展开。
3. IsaacSim owner leaf 声明 UniSim `entity_assets` 与 object
   `EntityVariantBinding`，mapped worker 负责 entity compose、MJCF→USD 转换、
   prototype 物化和 PhysX 步进。root defaults 与该默认 leaf 一致：
   `SimToolReal` + `sim_backend=isaacsim`。
4. MuJoCo owner leaf 消费同一套 entity 声明，由 UniSim compose 场景并把
   same-layout 工具池编译为 VariantPack，再交给 mjbatch CPU runtime。
5. Headless teacher line 不物化 GoalViz：权威 goal 在 command hub，观测/奖励/
   termination 都读 hub；因此不会为 1,200 个工具再构造 collision-disabled mirror
   或第二套 prototype。render/debug 标记以后必须走 visual-only 通道，不能重新
   引入完整工具实体。
