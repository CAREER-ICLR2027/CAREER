"""Paper constants and explicit run settings omitted from the manuscript."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class SearchConfig:
    global_confidence: float
    global_margin: float
    planning_tokens: int
    navigation_tokens: int
    verifier_tokens: int
    max_observations: int = 8
    absolute_support: float = 0.65
    relative_margin: float = 0.15
    support_improvement: float = 0.01
    margin_improvement: float = 0.01
    stagnation_patience: int = 2

    def __post_init__(self):
        for name in ("global_confidence", "global_margin", "absolute_support", "relative_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in ("planning_tokens", "navigation_tokens", "verifier_tokens", "stagnation_patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_observations) is not int or self.max_observations < 0:
            raise ValueError("max_observations must be a nonnegative integer")
        for name in ("support_improvement", "margin_improvement"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class RuntimeConfig:
    search: SearchConfig
    edge_size: tuple[int, int]
    edge_interpolation: str
    sgap: dict
    generator_context_tokens: int
    verifier_context_tokens: int

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeConfig":
        data = json.loads(Path(path).read_text())
        required = {
            "global_confidence", "global_margin", "planning_tokens", "navigation_tokens",
            "verifier_tokens", "edge_size", "edge_interpolation", "sgap",
            "generator_context_tokens", "verifier_context_tokens",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError(f"runtime config requires exactly: {sorted(required)}")
        search = SearchConfig(**{key: data[key] for key in (
            "global_confidence", "global_margin", "planning_tokens", "navigation_tokens", "verifier_tokens",
        )})
        size = data["edge_size"]
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 3 for v in size):
            raise ValueError("edge_size must be [height, width] with both at least 3")
        if data["edge_interpolation"] not in {"bilinear", "bicubic", "lanczos"}:
            raise ValueError("edge_interpolation must be bilinear, bicubic or lanczos")
        for key in ("generator_context_tokens", "verifier_context_tokens"):
            if type(data[key]) is not int or data[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        sgap = data["sgap"]
        keys = {"n_atoms", "pos_weight", "split_threshold", "keep_threshold", "use_local_normalization", "use_silhouette_score",
                "max_depth", "min_splits", "max_splits", "min_region_size", "max_nodes"}
        if not isinstance(sgap, dict) or set(sgap) != keys:
            raise ValueError(f"sgap requires exactly: {sorted(keys)}")
        for key in ("n_atoms", "max_depth", "min_splits", "max_splits", "min_region_size", "max_nodes"):
            if type(sgap[key]) is not int or sgap[key] < 1:
                raise ValueError(f"sgap.{key} must be a positive integer")
        if sgap["min_splits"] < 2 or sgap["max_splits"] < sgap["min_splits"]:
            raise ValueError("SGAP requires 2 <= min_splits <= max_splits")
        for key in ("use_local_normalization", "use_silhouette_score"):
            if type(sgap[key]) is not bool:
                raise ValueError(f"sgap.{key} must be boolean")
        for key in keys - {"n_atoms", "max_depth", "min_splits", "max_splits", "min_region_size", "max_nodes",
                           "use_local_normalization", "use_silhouette_score"}:
            value = sgap[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"sgap.{key} must be finite and nonnegative")
        return cls(search, tuple(size), data["edge_interpolation"], dict(sgap),
                   data["generator_context_tokens"], data["verifier_context_tokens"])
