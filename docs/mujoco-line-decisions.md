# MuJoCo Line Semantic Decisions (D1–D7)

> Filed on the `feat/mujoco-mjcf` branch as part of milestone M-J0.
> Source of truth: `mujoco-migration-roadmap.md` §2 in the workspace root.
> These decisions govern the MuJoCo/MJCF migration line only; the IsaacSim
> line on `main` is frozen and untouched.
>
> 2026-09-23: historical record. The MJCF dual-backend line has since become
> `main`, the pins below are superseded by `pyproject.toml`
> (`tool.uv.sources`), and the legacy IsaacSim URDF line it refers to is
> retired (frozen at tag `urdf-anchor`).

| # | Decision | Rationale |
|---|---|---|
| D1 | Goal is a cross-backend bring-up validation, not a training mainline and not label-by-label parity | Scope control; acceptance = builds, trains, sane metrics |
| D2 | Assets are emitted directly by the MJCF writer; robot/table are one-shot converted and checked in; mass/inertia/gains/friction live inside the assets | Single source of truth; MuJoCo consumes these semantics natively, no bake layer needed |
| D3 | Scene is a single composite MJCF (robot+table+tool in one file); variants go through the upstream model-level channel (paved by UniLab #1568) | Avoids the unlanded entity-level contract; a MuJoCo world is naturally a single model |
| D4 | goalviz v1 is not declared; the command term skips sim writes | Pure visualization twin; the hub buffer is authoritative; zero training impact |
| D5 | Variant masses come from a task-side mass table feeding wrench DR (`writer_masses[assignment]`), plus a 1200-variant compile-time self-check at asset generation | The MuJoCo compiler is the authority on its own format; no converter trust gap, no runtime facade needed |
| D6 | Gains/friction are carried natively by MJCF (`<position>` actuators, geom friction); owner override declarations are IsaacSim-line only | The MjcfConverter/URDF-importer semantic gaps do not exist on the MuJoCo line |
| D7 | Zero new upstream interfaces; a confirmed gap → upstream issue + task-side workaround | See upstream-dev-workflow.md §3 loop |

## Pin baseline for this line

- `unilab` = official UniLab `main` @ `2e3f30ae3f38e34b515e3ba6cf3db20780471ef6` (git rev pin)
- `unisim-core` = official PyPI release `>=1.4.2` (fork git source removed)
- Everything else unchanged (torch cu128 override, vendored RL-Games, indexes)
