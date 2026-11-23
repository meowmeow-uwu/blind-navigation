"""Model registry: reads configs/models.yaml and returns model metadata.

This module must stay free of heavy dependencies (no Ultralytics / torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


class ModelRegistryError(Exception):
    """Base error for registry problems."""


class UnknownModelError(ModelRegistryError, KeyError):
    """Requested model name is not defined in the config."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0])


class DisabledModelError(ModelRegistryError):
    """Requested model exists but is disabled in the config."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    role: str
    input_size: int
    task: str
    enabled: bool
    batch_size: int = 1
    device: str = "cpu"


class ModelRegistry:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        if not self.config_path.is_file():
            raise ModelRegistryError(f"Model config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        defaults = raw.get("defaults") or {}
        models = raw.get("models") or {}
        if not isinstance(models, dict) or not models:
            raise ModelRegistryError(f"No 'models' section in {self.config_path}")

        self._defaults = defaults
        self._specs: dict[str, ModelSpec] = {}
        for name, entry in models.items():
            entry = entry or {}
            if "checkpoint" not in entry:
                raise ModelRegistryError(
                    f"Model '{name}' in {self.config_path} is missing 'checkpoint'"
                )
            self._specs[name] = ModelSpec(
                name=name,
                checkpoint=entry["checkpoint"],
                role=entry.get("role", "unspecified"),
                input_size=int(entry.get("input_size", defaults.get("input_size", 320))),
                task=entry.get("task", defaults.get("task", "detect")),
                enabled=bool(entry.get("enabled", True)),
                batch_size=int(entry.get("batch_size", defaults.get("batch_size", 1))),
                device=str(entry.get("device", defaults.get("device", "cpu"))),
            )

    @property
    def defaults(self) -> dict:
        return dict(self._defaults)

    def names(self, include_disabled: bool = False) -> list[str]:
        return [
            n for n, s in self._specs.items() if include_disabled or s.enabled
        ]

    def get(self, name: str, include_disabled: bool = False) -> ModelSpec:
        spec = self._specs.get(name)
        if spec is None:
            known = ", ".join(self._specs) or "<none>"
            raise UnknownModelError(
                f"Unknown model '{name}'. Models defined in {self.config_path.name}: {known}"
            )
        if not spec.enabled and not include_disabled:
            raise DisabledModelError(
                f"Model '{name}' is disabled in {self.config_path.name} "
                "(set enabled: true or pass include_disabled=True)"
            )
        return spec
