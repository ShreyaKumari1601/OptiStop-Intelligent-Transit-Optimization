"""
utils/helpers.py
=================
Small, dependency-light utility functions shared across the pipeline:
logging setup, geospatial math, and directory bootstrapping.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Union

import numpy as np

EARTH_RADIUS_KM: float = 6371.0088


def setup_logger(name: str = "optistop", level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a configured logger with a consistent format.

    Args:
        name: Logger name. Reusing the same name returns the same logger
            instance instead of attaching duplicate handlers.
        level: Logging verbosity level (e.g. logging.DEBUG, logging.INFO).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def ensure_directories(*paths: Union[str, Path]) -> None:
    """Create directories (including parents) if they do not already exist.

    Args:
        *paths: One or more directory paths to guarantee exist on disk.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def haversine_km(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute the great-circle distance between two points in kilometers.

    Vectorized: any of the four arguments may be NumPy arrays, in which
    case the computation broadcasts element-wise.

    Args:
        lat1: Latitude of the first point(s), in decimal degrees.
        lon1: Longitude of the first point(s), in decimal degrees.
        lat2: Latitude of the second point(s), in decimal degrees.
        lon2: Longitude of the second point(s), in decimal degrees.

    Returns:
        Distance in kilometers (float or array, matching input shape).
    """
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), -1, 1))

    return EARTH_RADIUS_KM * c


def km_to_deg_lat(km: float) -> float:
    """Convert a north-south distance in kilometers to degrees of latitude."""
    return km / 111.32


def km_to_deg_lon(km: float, at_lat: float) -> float:
    """Convert an east-west distance in kilometers to degrees of longitude.

    Longitude degrees shrink in real-world distance as you move away from
    the equator, so the conversion depends on the latitude at which the
    measurement is taken.

    Args:
        km: Distance in kilometers.
        at_lat: Latitude (in decimal degrees) at which to evaluate the
            longitude-degree size.
    """
    return km / (111.32 * np.cos(np.radians(at_lat)))


def nearest_distance_km(
    points_lat: np.ndarray,
    points_lon: np.ndarray,
    targets_lat: Iterable[float],
    targets_lon: Iterable[float],
) -> np.ndarray:
    """For each point, compute distance (km) to the nearest of several targets.

    Args:
        points_lat: Array of latitudes for the source points.
        points_lon: Array of longitudes for the source points.
        targets_lat: Latitudes of candidate target locations (e.g. stops).
        targets_lon: Longitudes of candidate target locations.

    Returns:
        Array (same length as points) with the minimum distance, in km,
        from each point to any target location.
    """
    targets_lat = np.asarray(list(targets_lat))
    targets_lon = np.asarray(list(targets_lon))

    distances = np.stack(
        [
            haversine_km(points_lat, points_lon, t_lat, t_lon)
            for t_lat, t_lon in zip(targets_lat, targets_lon)
        ],
        axis=1,
    )
    return distances.min(axis=1)
