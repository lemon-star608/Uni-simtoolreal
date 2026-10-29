# DexToolBench MuJoCo evaluation

DexToolBench evaluates a SimToolReal policy in 24 committed static MJCF scenes. Each scene combines the KUKA iiwa14 and SHARPA robot with one fixed object layout and task trajectory. The evaluation environment runs one MuJoCo environment at a time and does not use the procedural SimToolReal tool pool.

## Task catalog

The catalog contains six categories and 24 unique category/object/task combinations:

- Hammer: claw and mallet
- Marker: Sharpie and Staples
- Eraser: handle eraser
- Brush: blue and red brushes
- Spatula: flat and spoon spatulas
- Screwdriver: long and short screwdrivers

Every task has:

- a committed full-scene MJCF;
- an upstream trajectory with start and goal poses;
- fixed no-randomization evaluation settings;
- the 29-joint robot layout and `startArmHigher` keyframe.

## Interactive inspection

```bash
uv run simtoolreal-dextoolbench-interactive
```

The launcher starts the MuJoCo worker, renders the selected scene in Viser, and executes zero actions. Category, object, and task controls select any of the 24 combinations.

## Policy evaluation

```bash
uv run simtoolreal-dextoolbench-interactive \
  --run logs/rlgames_sapg/SimToolReal/<run> \
  --checkpoint last
```

`--checkpoint` accepts `last` or `best` and resolves the matching file inside the run directory. The policy loader reads the run's persisted `run_config.json`, unwraps its `run`/`config` envelope, reads `rl_games.params`, and restores the selected checkpoint on CPU. The player adapts the metadata to one environment and pins the deterministic SAPG exploration coefficient block.

## Assets

`src/simtoolreal_unilab/assets/dextoolbench/` contains the committed scenes, trajectories, object meshes, shared robot/table meshes, UI image, license, and manifest. `uv run simtoolreal-assets` verifies all 312 entries by SHA-256.
