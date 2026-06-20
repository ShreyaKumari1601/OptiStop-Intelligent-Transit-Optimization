"""
config.py
=========
Centralized, single-source-of-truth configuration for the OptiStop pipeline.

Keeping every tunable parameter in one dataclass means the rest of the
codebase never hard-codes a "magic number" -- swap a city, change the
demand-point count, or widen the cluster search range from one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HotspotConfig:
    """A single real-world demand hotspot used to seed synthetic commuters.

    Attributes:
        name: Human-readable area name (used in plots/logs).
        lat: Latitude of the hotspot center.
        lon: Longitude of the hotspot center.
        weight: Relative share of total commuters generated around this
            hotspot (weights across all hotspots should sum to ~1.0).
        spread_km: Approximate standard deviation, in kilometers, of the
            Gaussian scatter of commuters around this hotspot. Larger
            values simulate more diffuse demand (e.g. a sprawling tech
            park vs. a dense market area).
    """

    name: str
    lat: float
    lon: float
    weight: float
    spread_km: float


@dataclass(frozen=True)
class Config:
    """Project-wide configuration.

    All paths are resolved relative to the project root so the pipeline
    behaves identically regardless of the working directory it is
    launched from.
    """

    # ---- Reproducibility -------------------------------------------------
    random_seed: int = 42

    # ---- Geography ---------------------------------------------------
    city_name: str = "Bangalore"
    city_center: Tuple[float, float] = (12.9716, 77.5946)  # (lat, lon)
    map_default_zoom: int = 12

    # ---- Synthetic demand generation -------------------------------------
    n_commuters: int = 4000
    noise_fraction: float = 0.05  # fraction of fully random "background" demand

    hotspots: List[HotspotConfig] = field(
        default_factory=lambda: [
            HotspotConfig("Koramangala", 12.9352, 77.6245, 0.12, 0.9),
            HotspotConfig("Indiranagar", 12.9719, 77.6412, 0.10, 0.8),
            HotspotConfig("MG Road / Trinity", 12.9758, 77.6045, 0.11, 0.6),
            HotspotConfig("Whitefield", 12.9698, 77.7500, 0.13, 1.4),
            HotspotConfig("Electronic City", 12.8452, 77.6602, 0.12, 1.3),
            HotspotConfig("Hebbal", 13.0358, 77.5970, 0.08, 1.0),
            HotspotConfig("Jayanagar", 12.9250, 77.5938, 0.09, 0.9),
            HotspotConfig("Yeshwanthpur", 13.0284, 77.5546, 0.07, 1.0),
            HotspotConfig("HSR Layout", 12.9116, 77.6389, 0.09, 0.9),
            HotspotConfig("Banashankari", 12.9255, 77.5468, 0.09, 1.0),
        ]
    )

    # ---- Clustering --------------------------------------------------
    k_min: int = 2
    k_max: int = 12
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300

    # ---- Coverage analytics -----------------------------------------
    acceptable_walk_km: float = 0.4  # ~5 minute walk, used for coverage stats

    # ---- Paths -------------------------------------------------------
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    outputs_dir: Path = project_root / "outputs"

    raw_data_filename: str = "synthetic_commuter_demand.csv"
    cluster_summary_filename: str = "cluster_summary.csv"
    evaluation_metrics_filename: str = "evaluation_metrics.txt"
    optimized_stops_map_filename: str = "optimized_stops_map.html"
    elbow_plot_filename: str = "elbow_curve.png"
    silhouette_plot_filename: str = "silhouette_scores.png"
    cluster_plot_filename: str = "cluster_plot.png"
    heatmap_plot_filename: str = "demand_heatmap.png"

    def hotspot_weights_sum_to_one(self) -> bool:
        """Sanity check that hotspot weights are a valid probability mass."""
        total = sum(h.weight for h in self.hotspots)
        return abs(total - 1.0) < 1e-6

    def as_dict(self) -> Dict:
        """Return a JSON-serializable summary of key config values (for logs)."""
        return {
            "city_name": self.city_name,
            "city_center": self.city_center,
            "n_commuters": self.n_commuters,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "random_seed": self.random_seed,
        }


CONFIG = Config()
