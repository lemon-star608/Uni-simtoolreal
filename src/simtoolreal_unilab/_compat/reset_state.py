"""Multi-rigid-root reset routing for ResetStateTransaction.

Method bodies below are ported verbatim from the SimToolReal development
branch of UniLab and applied onto the pinned official classes at import time
(see simtoolreal_unilab._compat). They exec() inside a copy of the official
module namespace so module-level helpers resolve exactly as upstream wrote
them.
"""

from __future__ import annotations

import textwrap
import time  # noqa: F401  (referenced by embedded sources)

import numpy as np
from unilab.base import reset_state as _module

_SOURCES = {
    "ResetStateTransaction.bind_root_state": (
        "    def bind_root_state(\n"
        "        self,\n"
        "        entity_name: str,\n"
        "        layout: BackendRootStateLayout,\n"
        "        default_root_state: np.ndarray,\n"
        "    ) -> None:\n"
        '        """Bind one entity root\'s independent state buffer on the cold path.\n'
        "\n"
        "        Articulation roots continue to use the generalized qpos/qvel buffers.\n"
        "        Only names advertised by the backend as rigid roots get a separate\n"
        "        13-D world-frame state buffer and are included in ``entity_root_states``\n"
        "        on commit.\n"
        '        """\n'
        "        del layout  # The rigid-slot contract is always the public 13-D layout.\n"
        "        # ManagerBasedRlEnv constructs this transaction before EntityScene\n"
        "        # resolves backend metadata.  Refresh once here after the first entity\n"
        "        # lookup has materialized the worker, otherwise rigid roots would be\n"
        "        # mistaken for articulation qpos/qvel columns.\n"
        "        if not self._rigid_root_entities:\n"
        "            try:\n"
        "                names = self._backend.get_rigid_root_entities()\n"
        "            except (AttributeError, NotImplementedError):\n"
        "                names = ()\n"
        "            if names:\n"
        "                self._rigid_root_entities = frozenset(str(name) for name in names)\n"
        "                self._entity_root_dirty_masks = {\n"
        "                    name: np.zeros(self._num_envs, dtype=np.bool_)\n"
        "                    for name in self._rigid_root_entities\n"
        "                }\n"
        "        name = str(entity_name)\n"
        "        if name not in self._rigid_root_entities:\n"
        "            return\n"
        "        values = np.asarray(default_root_state)\n"
        "        if values.shape != (self._num_envs, 13):\n"
        "            raise ValueError(\n"
        "                f\"Entity '{name}' default root state has shape {values.shape}; \"\n"
        '                f"expected ({self._num_envs}, 13)"\n'
        "            )\n"
        "        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():\n"
        "            raise TypeError(f\"Entity '{name}' default root state must be finite floating data\")\n"
        "        previous = self._root_defaults.get(name)\n"
        "        if previous is not None and not np.array_equal(previous, values):\n"
        "            raise ValueError(f\"Entity '{name}' root state was bound with conflicting defaults\")\n"
        "        if previous is None:\n"
        "            default = np.array(values, copy=True)\n"
        "            default.setflags(write=False)\n"
        "            self._root_defaults[name] = default\n"
        "            self._entity_root_states[name] = np.empty_like(default)\n"
    ),
    "ResetStateTransaction.write_root_state": (
        "    def write_root_state(\n"
        "        self,\n"
        "        env_ids: np.ndarray,\n"
        "        layout: BackendRootStateLayout,\n"
        "        root_state: np.ndarray,\n"
        "        *,\n"
        "        entity_name: str | None = None,\n"
        "        term_name: str,\n"
        "    ) -> None:\n"
        '        """Stage a community 13-D world-frame root state."""\n'
        "        self._require_active()\n"
        '        ids = self._validate_ids(env_ids, capability="write_root_state")\n'
        "        values = self._validate_values(\n"
        "            root_state,\n"
        "            shape=(ids.size, 13),\n"
        '            capability="root state",\n'
        "            term_name=term_name,\n"
        "        )\n"
        "        self.write_root_pose(\n"
        "            ids,\n"
        "            layout,\n"
        "            values[:, :7],\n"
        "            entity_name=entity_name,\n"
        "            term_name=term_name,\n"
        "        )\n"
        "        self.write_root_velocity(\n"
        "            ids,\n"
        "            layout,\n"
        "            values[:, 7:],\n"
        "            entity_name=entity_name,\n"
        "            term_name=term_name,\n"
        "        )\n"
    ),
    "ResetStateTransaction.write_root_pose": (
        "    def write_root_pose(\n"
        "        self,\n"
        "        env_ids: np.ndarray,\n"
        "        layout: BackendRootStateLayout,\n"
        "        pose: np.ndarray,\n"
        "        *,\n"
        "        entity_name: str | None = None,\n"
        "        term_name: str,\n"
        "    ) -> None:\n"
        '        """Stage world position and wxyz orientation for one floating root."""\n'
        "        if entity_name is not None and str(entity_name) in self._rigid_root_entities:\n"
        "            ids = self._prepare_entity_root_write(\n"
        '                str(entity_name), env_ids, capability="root-pose", term_name=term_name\n'
        "            )\n"
        "            positions = self._validate_values(\n"
        "                pose,\n"
        "                shape=(ids.size, 7),\n"
        '                capability="root pose",\n'
        "                term_name=term_name,\n"
        "            )\n"
        "            self._validate_quaternions(positions[:, 3:7], term_name=term_name)\n"
        "            if ids.size:\n"
        "                self._entity_root_states[str(entity_name)][ids, :7] = positions\n"
        "                self._entity_root_dirty_masks[str(entity_name)][ids] = True\n"
        "            return\n"
        '        ids = self._prepare_state_write(env_ids, capability="root-pose", term_name=term_name)\n'
        "        positions = self._validate_values(\n"
        "            pose,\n"
        "            shape=(ids.size, 7),\n"
        '            capability="root pose",\n'
        "            term_name=term_name,\n"
        "        )\n"
        "        self._validate_quaternions(positions[:, 3:7], term_name=term_name)\n"
        "        qpos_columns, _ = self._validate_root_layout(layout, term_name=term_name)\n"
        "        assert self._qpos is not None\n"
        "        if ids.size:\n"
        "            self._qpos[ids[:, None], qpos_columns[None, :]] = positions\n"
        "        self._dirty_mask[ids] = True\n"
    ),
    "ResetStateTransaction.write_root_velocity": (
        "    def write_root_velocity(\n"
        "        self,\n"
        "        env_ids: np.ndarray,\n"
        "        layout: BackendRootStateLayout,\n"
        "        velocity_w: np.ndarray,\n"
        "        *,\n"
        "        entity_name: str | None = None,\n"
        "        term_name: str,\n"
        "    ) -> None:\n"
        '        """Stage world-frame root velocity in generalized qvel columns.\n'
        "\n"
        "        The public generalized-state contract stores free-root linear velocity\n"
        "        in world coordinates and angular velocity in root-body coordinates.\n"
        "        The conversion uses the pose already staged in this transaction.\n"
        '        """\n'
        "        if entity_name is not None and str(entity_name) in self._rigid_root_entities:\n"
        "            ids = self._prepare_entity_root_write(\n"
        '                str(entity_name), env_ids, capability="root-velocity", term_name=term_name\n'
        "            )\n"
        "            velocities = self._validate_values(\n"
        "                velocity_w,\n"
        "                shape=(ids.size, 6),\n"
        '                capability="root velocity",\n'
        "                term_name=term_name,\n"
        "            )\n"
        "            if ids.size:\n"
        "                # Independent rigid root slots use world-frame angular\n"
        "                # velocity, matching the public entity_root_states contract.\n"
        "                self._entity_root_states[str(entity_name)][ids, 7:13] = velocities\n"
        "                self._entity_root_dirty_masks[str(entity_name)][ids] = True\n"
        "            return\n"
        '        ids = self._prepare_state_write(env_ids, capability="root-velocity", term_name=term_name)\n'
        "        velocities = self._validate_values(\n"
        "            velocity_w,\n"
        "            shape=(ids.size, 6),\n"
        '            capability="root velocity",\n'
        "            term_name=term_name,\n"
        "        )\n"
        "        qpos_columns, qvel_columns = self._validate_root_layout(layout, term_name=term_name)\n"
        "        assert self._qpos is not None\n"
        "        assert self._qvel is not None\n"
        "        if ids.size:\n"
        "            quaternions = self._qpos[\n"
        "                ids[:, None],\n"
        "                qpos_columns[None, 3:7],\n"
        "            ]\n"
        "            self._validate_quaternions(quaternions, term_name=term_name)\n"
        "            encoded_velocity = np.array(velocities, copy=True)\n"
        "            encoded_velocity[:, 3:6] = np_quat_apply_inverse(\n"
        "                quaternions,\n"
        "                velocities[:, 3:6],\n"
        "            )\n"
        "            self._qvel[ids[:, None], qvel_columns[None, :]] = encoded_velocity\n"
        "        self._dirty_mask[ids] = True\n"
    ),
    "ResetStateTransaction.read_root_pose": (
        "    def read_root_pose(\n"
        "        self,\n"
        "        env_ids: np.ndarray,\n"
        "        layout: BackendRootStateLayout,\n"
        "        *,\n"
        "        entity_name: str | None = None,\n"
        "        term_name: str,\n"
        "    ) -> np.ndarray:\n"
        '        """Read the world position and wxyz orientation staged for one floating root.\n'
        "\n"
        "        Returns a detached ``(len(env_ids), 7)`` copy of the pose currently\n"
        "        staged in this transaction. Rows no term has written yet are first\n"
        "        initialized to the backend default pose, so a later reset term can\n"
        "        build on an earlier term's root placement without re-deriving it.\n"
        '        """\n'
        "        if entity_name is not None and str(entity_name) in self._rigid_root_entities:\n"
        "            ids = self._prepare_entity_root_write(\n"
        '                str(entity_name), env_ids, capability="root-pose", term_name=term_name\n'
        "            )\n"
        "            return np.array(self._entity_root_states[str(entity_name)][ids, :7], copy=True)\n"
        '        ids = self._prepare_state_write(env_ids, capability="root-pose", term_name=term_name)\n'
        "        qpos_columns, _ = self._validate_root_layout(layout, term_name=term_name)\n"
        "        assert self._qpos is not None\n"
        "        return np.array(self._qpos[ids[:, None], qpos_columns[None, :]], copy=True)\n"
    ),
    "ResetStateTransaction.commit": (
        "    def commit(self) -> dict | None:\n"
        '        """Commit all staged rows through one public backend call."""\n'
        "        self._require_active()\n"
        "        entity_dirty = self._entity_dirty_scratch\n"
        "        entity_dirty.fill(False)\n"
        "        for mask in self._entity_root_dirty_masks.values():\n"
        "            entity_dirty |= mask\n"
        "        dirty_ids = np.flatnonzero(self._dirty_mask | entity_dirty).astype(np.int32, copy=False)\n"
        "        mocap_dirty = any(np.any(mask) for mask in self._mocap_masks.values())\n"
        "        self._last_commit_had_writes = bool(dirty_ids.size) or mocap_dirty\n"
        "        try:\n"
        "            if dirty_ids.size == 0:\n"
        "                self._commit_mocap_poses()\n"
        "                return None\n"
        "            randomization = self._build_randomization_payload(dirty_ids)\n"
        "            robot_write = bool(np.any(self._dirty_mask[dirty_ids]))\n"
        "            qpos: np.ndarray | None = None\n"
        "            qvel: np.ndarray | None = None\n"
        "            if robot_write:\n"
        "                assert self._qpos is not None\n"
        "                assert self._qvel is not None\n"
        "                qpos = self._qpos[dirty_ids]\n"
        "                qvel = self._qvel[dirty_ids]\n"
        "            entity_root_states: dict[str, np.ndarray] = {}\n"
        "            for name, mask in self._entity_root_dirty_masks.items():\n"
        "                if not np.any(mask[dirty_ids]):\n"
        "                    continue\n"
        "                if not np.all(mask[dirty_ids]):\n"
        "                    raise RuntimeError(\n"
        '                        "One reset transaction cannot represent sparse writes for rigid "\n'
        "                        f\"entity '{name}' across rows {dirty_ids.tolist()}\"\n"
        "                    )\n"
        "                entity_root_states[name] = np.array(\n"
        "                    self._entity_root_states[name][dirty_ids], copy=True\n"
        "                )\n"
        "            try:\n"
        "                set_state_t0 = time.perf_counter()\n"
        '                kwargs: dict[str, object] = {"randomization": randomization}\n'
        "                if entity_root_states:\n"
        '                    kwargs["entity_root_states"] = entity_root_states\n'
        "                result = self._backend.set_state(dirty_ids, qpos, qvel, **kwargs)\n"
        "                self._record_committed_payload(dirty_ids, randomization)\n"
        "                self._commit_mocap_poses()\n"
        "                timing: dict[str, float] = {\n"
        '                    "dr_reset_set_state_ms": (time.perf_counter() - set_state_t0) * 1000.0\n'
        "                }\n"
        "                if isinstance(result, dict):\n"
        '                    backend_timing = result.get("timing")\n'
        "                    if isinstance(backend_timing, dict):\n"
        "                        timing.update(backend_timing)\n"
        "                self._last_set_state_timing_ms = timing\n"
        "                return cast(dict | None, result)\n"
        "            except (AttributeError, NotImplementedError) as exc:\n"
        '                terms = ", ".join(sorted(self._requesting_terms))\n'
        "                raise NotImplementedError(\n"
        "                    \"EventManager reset-state capability 'SimBackend.set_state' is unavailable \"\n"
        "                    f\"for term(s) [{terms}] on backend '{self._backend.backend_type}': {exc}\"\n"
        "                ) from exc\n"
        "        finally:\n"
        "            self._finish()\n"
    ),
    "ResetStateTransaction._prepare_entity_root_write": (
        "    def _prepare_entity_root_write(\n"
        "        self,\n"
        "        entity_name: str,\n"
        "        env_ids: np.ndarray,\n"
        "        *,\n"
        "        capability: str,\n"
        "        term_name: str,\n"
        "    ) -> np.ndarray:\n"
        "        self._require_active()\n"
        '        ids = self._validate_ids(env_ids, capability=f"write_{capability}")\n'
        "        outside = ids[~self._active_mask[ids]]\n"
        "        if outside.size:\n"
        "            raise ValueError(\n"
        "                f\"EventManager term '{term_name}' attempted {capability} mutation outside \"\n"
        '                f"the active reset: {outside.tolist()}"\n'
        "            )\n"
        "        try:\n"
        "            defaults = self._root_defaults[entity_name]\n"
        "            states = self._entity_root_states[entity_name]\n"
        "            dirty = self._entity_root_dirty_masks[entity_name]\n"
        "        except KeyError as exc:\n"
        "            raise RuntimeError(\n"
        "                f\"Entity root '{entity_name}' was not bound during scene materialization\"\n"
        "            ) from exc\n"
        "        self._requesting_terms.add(term_name)\n"
        "        uninitialized = ids[~dirty[ids]]\n"
        "        if uninitialized.size:\n"
        "            states[uninitialized] = defaults[uninitialized]\n"
        "        return ids\n"
    ),
}


for _name in list(_SOURCES):
    _SOURCES[_name] = textwrap.dedent(_SOURCES[_name])


def apply() -> None:
    exec_ns = vars(_module)
    for name, source in _SOURCES.items():
        exec(compile(source, f"<compat.reset_state.{name}>", "exec"), exec_ns)
    cls = _module.ResetStateTransaction
    for name, source in _SOURCES.items():
        setattr(cls, name.split(".")[-1], exec_ns[name.split(".")[-1]])

    orig_init = cls.__init__

    def _init(self, backend, *args, **kwargs):
        orig_init(self, backend, *args, **kwargs)
        try:
            rigid_entities = backend.get_rigid_root_entities()
        except (AttributeError, NotImplementedError):
            rigid_entities = ()
        self._rigid_root_entities = frozenset(str(name) for name in rigid_entities)
        self._root_defaults: dict[str, np.ndarray] = {}
        self._entity_root_states: dict[str, np.ndarray] = {}
        self._entity_root_dirty_masks: dict[str, np.ndarray] = {
            name: np.zeros(self._num_envs, dtype=np.bool_) for name in self._rigid_root_entities
        }
        self._entity_dirty_scratch = np.zeros(self._num_envs, dtype=np.bool_)

    cls.__init__ = _init

    orig_begin = cls.begin

    def _begin(self, env_ids):
        orig_begin(self, env_ids)
        for mask in self._entity_root_dirty_masks.values():
            mask.fill(False)

    cls.begin = _begin
