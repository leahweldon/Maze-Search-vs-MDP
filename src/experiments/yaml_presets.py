from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    mode: str  # "full" or "search_only"
    maze_type: str
    sizes: List[int]
    seeds: List[int]
    out_csv: str
    mazes_dir: str = "results/mazes"

    # full mode
    gammas: Optional[List[float]] = None
    theta: float = 1e-6

    # random only
    density: float = 0.25

    # search_only only
    track_peak_mem: bool = False


def load_presets(yaml_path: str) -> Dict[str, ExperimentPreset]:
    path = Path(yaml_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    presets_raw: Dict[str, Any] = data.get("presets", {})
    presets: Dict[str, ExperimentPreset] = {}

    for name, cfg in presets_raw.items():
        presets[name] = ExperimentPreset(
            name=name,
            mode=cfg["mode"],
            maze_type=cfg["maze_type"],
            sizes=list(cfg["sizes"]),
            seeds=list(cfg["seeds"]),
            out_csv=cfg["out_csv"],
            mazes_dir=cfg.get("mazes_dir", "results/mazes"),
            gammas=list(cfg["gammas"]) if "gammas" in cfg else None,
            theta=float(cfg.get("theta", 1e-6)),
            density=float(cfg.get("density", 0.25)),
            track_peak_mem=bool(cfg.get("track_peak_mem", False)),
        )

    return presets
