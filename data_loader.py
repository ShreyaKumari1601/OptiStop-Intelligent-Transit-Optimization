"""
data_loader.py
===============
Synthetic commuter demand generation for the OptiStop pipeline.

Real commuter-origin data (e.g. ticketing or smart-card taps) is rarely
public, so this module builds a realistic stand-in: a mixture of Gaussian
clusters centered on well-known Bangalore demand areas, plus a thin layer
of uniform background noise to mimic stray/low-density demand.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.utils.helpers import ensure_directories, km_to_deg_lat, km_to_deg_lon

logger = logging.getLogger("optistop")


class CommuterDemandGenerator:
    """Generates synthetic geospatial commuter demand points.

    The generator places `n_commuters` points as a weighted mixture of
    Gaussian blobs around configured hotspots (e.g. Koramangala,
    Whitefield), with a small fraction scattered uniformly across the
    city's bounding box to simulate background demand.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the generator.

        Args:
            config: Project configuration containing hotspot definitions,
                city center, and the desired commuter count.

        Raises:
            ValueError: If hotspot weights do not sum to ~1.0, or if
                `n_commuters` is not a positive integer.
        """
        self.config = config

        if not config.hotspot_weights_sum_to_one():
            total = sum(h.weight for h in config.hotspots)
            raise ValueError(
                f"Hotspot weights must sum to ~1.0, got {total:.4f}. "
                "Check src/config.py HotspotConfig definitions."
            )

        if config.n_commuters <= 0:
            raise ValueError("n_commuters must be a positive integer.")

        self._rng = np.random.default_rng(config.random_seed)

    def generate(self) -> pd.DataFrame:
        """Generate the full synthetic commuter demand dataset.

        Returns:
            DataFrame with columns ``["latitude", "longitude", "demand_weight",
            "source_hotspot"]``, one row per synthetic commuter.
        """
        n_noise = int(round(self.config.n_commuters * self.config.noise_fraction))
        n_hotspot_points = self.config.n_commuters - n_noise

        logger.info(
            "Generating %d synthetic commuter points (%d hotspot-driven, %d background noise)",
            self.config.n_commuters,
            n_hotspot_points,
            n_noise,
        )

        hotspot_frames = [
            self._generate_for_hotspot(h, n_hotspot_points) for h in self.config.hotspots
        ]
        noise_frame = self._generate_background_noise(n_noise)

        df = pd.concat(hotspot_frames + [noise_frame], ignore_index=True)

        # Assign each commuter a demand weight (e.g. proxy for trips/day),
        # log-normal so most points are "typical" with some high-demand outliers.
        df["demand_weight"] = self._rng.lognormal(mean=0.0, sigma=0.5, size=len(df))
        df["demand_weight"] = df["demand_weight"].round(2)

        df = df.sample(frac=1.0, random_state=self.config.random_seed).reset_index(drop=True)
        logger.info("Generated dataframe with shape %s", df.shape)
        return df

    def _generate_for_hotspot(self, hotspot, n_hotspot_points: int) -> pd.DataFrame:
        """Generate Gaussian-scattered points around a single hotspot."""
        n_points = max(1, int(round(n_hotspot_points * hotspot.weight)))

        lat_std = km_to_deg_lat(hotspot.spread_km)
        lon_std = km_to_deg_lon(hotspot.spread_km, at_lat=hotspot.lat)

        lats = self._rng.normal(loc=hotspot.lat, scale=lat_std, size=n_points)
        lons = self._rng.normal(loc=hotspot.lon, scale=lon_std, size=n_points)

        return pd.DataFrame(
            {
                "latitude": lats,
                "longitude": lons,
                "source_hotspot": hotspot.name,
            }
        )

    def _generate_background_noise(self, n_noise: int) -> pd.DataFrame:
        """Generate uniformly scattered background demand across the city."""
        if n_noise <= 0:
            return pd.DataFrame(columns=["latitude", "longitude", "source_hotspot"])

        # ~18km bounding radius around city center covers greater Bangalore.
        radius_km = 18.0
        lat_span = km_to_deg_lat(radius_km)
        lon_span = km_to_deg_lon(radius_km, at_lat=self.config.city_center[0])

        lats = self._rng.uniform(
            self.config.city_center[0] - lat_span,
            self.config.city_center[0] + lat_span,
            size=n_noise,
        )
        lons = self._rng.uniform(
            self.config.city_center[1] - lon_span,
            self.config.city_center[1] + lon_span,
            size=n_noise,
        )
        return pd.DataFrame(
            {"latitude": lats, "longitude": lons, "source_hotspot": "background"}
        )


def save_demand_data(df: pd.DataFrame, path: Path) -> None:
    """Persist the generated demand dataset to CSV.

    Args:
        df: Commuter demand DataFrame.
        path: Destination CSV path. Parent directories are created
            automatically if missing.
    """
    ensure_directories(path.parent)
    df.to_csv(path, index=False)
    logger.info("Saved synthetic demand data to %s (%d rows)", path, len(df))


def load_demand_data(path: Path) -> pd.DataFrame:
    """Load a previously generated commuter demand dataset from CSV.

    Args:
        path: Path to the CSV file to load.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If no file exists at `path`.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No demand data found at {path}. Run the generator first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded demand data from %s (%d rows)", path, len(df))
    return df


def get_or_generate_demand_data(
    config: Config, force_regenerate: bool = False, save: bool = True
) -> pd.DataFrame:
    """Load cached demand data if present, otherwise generate fresh data.

    Args:
        config: Project configuration.
        force_regenerate: If True, ignore any cached CSV and regenerate.
        save: If True, persist freshly generated data to disk.

    Returns:
        Commuter demand DataFrame.
    """
    data_path = config.data_dir / config.raw_data_filename

    if not force_regenerate and data_path.exists():
        return load_demand_data(data_path)

    generator = CommuterDemandGenerator(config)
    df = generator.generate()

    if save:
        save_demand_data(df, data_path)

    return df
