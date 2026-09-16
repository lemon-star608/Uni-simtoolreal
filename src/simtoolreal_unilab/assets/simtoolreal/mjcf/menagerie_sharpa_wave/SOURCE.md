# Sharpa Wave Menagerie asset provenance

This directory vendors collision mesh pieces from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
under the Apache-2.0 license.

## Pinned revisions

- `google-deepmind/mujoco_menagerie` repository commit:
  [`da76818e269b82289eba39808e2fb91d679d6994`](https://github.com/google-deepmind/mujoco_menagerie/commit/da76818e269b82289eba39808e2fb91d679d6994)
- `google-deepmind/mujoco_menagerie` Sharpa directory commit:
  [`c1a4eeb85694ae1dffe33ff1797d4e528928a133`](https://github.com/google-deepmind/mujoco_menagerie/commit/c1a4eeb85694ae1dffe33ff1797d4e528928a133)
- Original `sharpa-robotics/sharpa-urdf-usd-xml` source commit:
  [`6eea427eb24189519f32b9f21674cd534d3f973c`](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/commit/6eea427eb24189519f32b9f21674cd534d3f973c)

The vendored OBJ bytes come from the exact upstream paths
`sharpa_wave/assets/left/palm/palm000.obj` through
`sharpa_wave/assets/left/palm/palm031.obj` at the pinned Menagerie repository
commit. The license bytes come from the exact upstream path
`sharpa_wave/LICENSE` at the same commit and are preserved locally as
[`LICENSE`](LICENSE).

## Collision-only transplant deviations

This is a collision-only transplant, not a complete unchanged Menagerie model.
UniLab body frames, visuals, inertia, joints, actuators, sensors, and contact
excludes stay authoritative. Menagerie geom sizes, poses, quaternions, and mesh
pieces are copied. UniLab friction, contact bits, and viewer groups remain
authoritative.
