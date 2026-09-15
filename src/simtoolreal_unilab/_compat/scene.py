"""SceneCfg entity-assets materialization applied to the official class.

Ported from the SimToolReal development branch of UniLab (unilab.base.scene
``SceneCfg.__post_init__`` extension). Wrapping ``__post_init__`` (rather
than subclassing) keeps every construction path covered — the generic config
machinery instantiates ``unilab.base.scene.SceneCfg`` directly from the owner
YAML tree.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from unilab.base import scene as _module
from unisim.scene import (
    ActuatorGainOverride,
    BodyFrictionOverride,
    SceneEntitySpec,
)


def apply() -> None:
    cls = _module.SceneCfg
    if "entity_assets" in inspect.getsource(cls.__post_init__):
        raise RuntimeError(
            "official UniLab SceneCfg already materializes entity_assets; the "
            "pinned revision drifted away from what simtoolreal_unilab._compat "
            "expects — revisit the compatibility layer"
        )
    orig_post_init = cls.__post_init__

    def _post_init(self) -> None:
        orig_post_init(self)
        # Hydra/OmegaConf hands tuple fields to the dataclass constructor as
        # a list of plain mappings. Materialize the backend-facing asset
        # declarations here so every backend receives the typed cold-path
        # contract, just like the manager-facing ``entities`` mapping.
        assets: list[SceneEntitySpec] = []
        for index, value in enumerate(self.entity_assets):
            if isinstance(value, SceneEntitySpec):
                assets.append(value)
            elif isinstance(value, Mapping):
                try:
                    raw: dict[str, Any] = dict(value)
                    for key, item_type in (
                        ("actuator_gain_overrides", ActuatorGainOverride),
                        ("contact_friction_by_body", BodyFrictionOverride),
                    ):
                        raw_values = raw.get(key, ())
                        converted = tuple(
                            item if isinstance(item, item_type) else item_type(**item)
                            for item in raw_values
                        )
                        raw[key] = converted
                    assets.append(SceneEntitySpec(**raw))
                except Exception as exc:
                    raise TypeError(
                        f"SceneCfg entity_assets[{index}] could not materialize "
                        f"SceneEntitySpec: {exc}"
                    ) from exc
            else:
                raise TypeError(
                    f"SceneCfg entity_assets[{index}] must be SceneEntitySpec or mapping, "
                    f"got {type(value).__name__}"
                )
        self.entity_assets = tuple(assets)

    cls.__post_init__ = _post_init
