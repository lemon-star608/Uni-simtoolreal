# Architecture

## SimToolReal teacher training

The training path is fixed to the `simtoolreal/mujoco` task:

1. The CLI and Hydra root select `DEFAULT_TASK = simtoolreal/mujoco`.
2. The SimToolReal owner composes task semantics, rewards, observations, commands, events, and terminations.
3. Committed robot/table sources and the deterministic tool catalog are serialized into entity-split MJCF files.
4. `SimToolRealEnvCfg.bind_entity_variant` assigns the 1,200 tool variants across 24,576 environments.
5. UniSim composes the entity sources into a mjbatch VariantPack.
6. UniLab TorchEnv exposes batched state, observations, rewards, and resets to the RL-Games adapter.
7. The vendored RL-Games SAPG runtime trains the teacher policy and saves checkpoints and TensorBoard events.

The factory binds the variant plan before constructing the environment. Reset writes use MuJoCo's packed state and kinematic table mocap channels, while the action term applies targets with the canonical joint-id tuple.

## DexToolBench evaluation

DexToolBench uses a separate direct-scene path:

1. The catalog resolves one of 24 committed static MJCF files and trajectories.
2. The task owner creates a single-environment `SceneCfg` with that file as `model_file`.
3. UniLab constructs the fixed MuJoCo environment without the SimToolReal tool-pool binding.
4. The worker exposes the observation/action contract to the interactive launcher.
5. Viser renders the scene, object, and goal; optionally, the SAPG player consumes a persisted SimToolReal run configuration and checkpoint.

This separation keeps procedural teacher training independent from the committed evaluation scenes.

## Package boundaries

- `tasks/simtoolreal` owns training semantics and procedural MJCF generation.
- `tasks/dextoolbench` owns catalog and fixed-scene configuration.
- `dextoolbench` owns worker, policy, rendering, and interactive process orchestration.
- `rlgames_sapg` owns the UniLab-to-RL-Games runtime adapter and checkpoint identity.
- `assets.py` verifies both committed asset manifests.
