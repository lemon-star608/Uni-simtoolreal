# DexToolBench on MuJoCo

This branch carries the upstream benchmark's 24 fixed combinations as a
separate UniLab task, **not** as another SimToolReal tool-pool variant:

- 6 categories × 2 objects × 2 tasks
- one committed, directly loaded MJCF per task
- one MuJoCo backend, one environment
- no runtime MJCF writer and no procedural tool pool
- the upstream interactive Viser launcher layout

## Run

Zero-action inspection (useful before supplying a checkpoint):

```bash
uv run simtoolreal-dextoolbench-interactive --port 8080
```

Policy demo:

```bash
uv run simtoolreal-dextoolbench-interactive \
  --config-path /path/to/config.yaml \
  --checkpoint-path /path/to/model.pth
```

Both the original upstream `train.params` policy YAML and this repository's
`rl_games.params` YAML schema are accepted. Playback is deterministic and pins
the SAPG exploration coefficient to the upstream eval value (`50.0`).

Select a category/object/task in Viser and click **Load Environment**. The
launcher starts a fresh MuJoCo worker, then **Run Episode** streams joint,
object, goal, progress, and success state at 60 Hz. Load another combination
kills and respawns the worker exactly like upstream.

## Alignment

The static scenes reproduce the upstream task inputs:

- robot pose, 29 joints/actuators, gains, contact exclusions, and solver profile
- DexToolBench `startArmHigher=True` keyframe/reset pose
- original object visual OBJs and offline CoACD collision parts
- explicit inertials from upstream `*_decomposed.urdf`
- task tables/props (whiteboard, nail, wall, or plain table)
- trajectory `start_pose` with `z += 0.03`
- fixed goal list and `max_consecutive_successes = len(goals)`
- normalized object scale in the actor observation (not a physical mesh scale)

Runtime randomization is disabled for evaluation: reset position/DOF/table
noise, object rotation, object-scale noise, observation/action delay, joint
velocity observation noise, and object wrench scales are all pinned off.
Success uses tolerance `0.01`, `successSteps=1`, and fixed-size keypoints as in
upstream DexToolBench.

Assets and trajectories are vendored from
`tylerlum/simtoolreal @ 313d5aea1f507c6cfe097b672b62945d7b0bbff5` (MIT) and
checked by `manifest.json`. Run `uv run simtoolreal-assets` to verify them.

## Current first-version boundary

- The Viser renderer consumes static MJCF visual geoms and uses texture-average
  material colors; full per-texel Viser texture mapping is a follow-up.
- Batch `eval.json` generation is not included yet; the interactive worker is
  the first runtime path.
- Physics is MuJoCo, so trajectories are task-aligned rather than expected to
  match IsaacGym/IsaacSim timestep-for-timestep.
