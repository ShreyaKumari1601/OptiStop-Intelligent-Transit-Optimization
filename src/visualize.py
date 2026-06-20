"""
visualize.py
=============
All plotting and map-generation for OptiStop:
- Elbow curve (Matplotlib)
- Silhouette score curve (Matplotlib)
- Cluster scatter plot (Matplotlib/Seaborn)
- Demand density heatmap (Seaborn)
- Interactive Folium map with demand heatmap + recommended stop markers
"""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap

from src.clustering import OptimalKResult
from src.utils.helpers import ensure_directories

logger = logging.getLogger("optistop")

sns.set_theme(style="whitegrid")


def plot_elbow_curve(result: OptimalKResult, save_path: Path) -> None:
    """Plot inertia vs. k (Elbow Method) and save to disk.

    Args:
        result: Output of `TransitStopOptimizer.find_optimal_k`.
        save_path: Destination PNG path.
    """
    ensure_directories(save_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.k_values, result.inertias, marker="o", color="#2563eb", linewidth=2)
    ax.axvline(result.best_k, color="#dc2626", linestyle="--", alpha=0.7, label=f"Selected k={result.best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow Method for Optimal k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved elbow curve to %s", save_path)


def plot_silhouette_scores(result: OptimalKResult, save_path: Path) -> None:
    """Plot silhouette score vs. k and save to disk.

    Args:
        result: Output of `TransitStopOptimizer.find_optimal_k`.
        save_path: Destination PNG path.
    """
    ensure_directories(save_path.parent)
    valid = [(k, s) for k, s in zip(result.k_values, result.silhouette_scores) if not np.isnan(s)]
    ks, scores = zip(*valid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, marker="o", color="#16a34a", linewidth=2)
    ax.axvline(result.best_k, color="#dc2626", linestyle="--", alpha=0.7, label=f"Selected k={result.best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by Number of Clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved silhouette score plot to %s", save_path)


def plot_clusters(
    df: pd.DataFrame, labels: np.ndarray, centers: np.ndarray, save_path: Path
) -> None:
    """Scatter-plot commuter points colored by cluster, with stop markers overlaid.

    Args:
        df: Demand DataFrame with `latitude`, `longitude`.
        labels: Cluster label per commuter.
        centers: Array of shape (k, 2) of recommended stop coordinates.
        save_path: Destination PNG path.
    """
    ensure_directories(save_path.parent)
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(
        df["longitude"],
        df["latitude"],
        c=labels,
        cmap="tab20",
        s=10,
        alpha=0.6,
        linewidths=0,
    )
    ax.scatter(
        centers[:, 1],
        centers[:, 0],
        c="red",
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.2,
        label="Recommended stop",
    )
    for i, (lat, lon) in enumerate(centers):
        ax.annotate(str(i), (lon, lat), fontsize=9, fontweight="bold", ha="center", va="center")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Commuter Demand Clusters (k={len(centers)}) & Recommended Stops")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved cluster plot to %s", save_path)


def plot_demand_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Plot a static 2D KDE heatmap of commuter demand density (Seaborn).

    Args:
        df: Demand DataFrame with `latitude`, `longitude`, `demand_weight`.
        save_path: Destination PNG path.
    """
    ensure_directories(save_path.parent)
    fig, ax = plt.subplots(figsize=(9, 8))

    sns.kdeplot(
        x=df["longitude"],
        y=df["latitude"],
        weights=df["demand_weight"],
        fill=True,
        cmap="rocket_r",
        thresh=0.02,
        levels=60,
        ax=ax,
    )
    ax.scatter(df["longitude"], df["latitude"], s=2, color="black", alpha=0.15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Commuter Demand Density Heatmap")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved demand heatmap to %s", save_path)


def build_interactive_map(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    city_center: tuple,
    zoom_start: int,
    save_path: Path,
) -> folium.Map:
    """Build an interactive Folium map with a demand heatmap layer and
    clickable markers for each recommended optimal stop.

    Args:
        df: Demand DataFrame with `latitude`, `longitude`, `demand_weight`.
        cluster_summary: Output of `analytics.build_cluster_summary`.
        city_center: (lat, lon) tuple used to center the map.
        zoom_start: Initial Folium zoom level.
        save_path: Destination HTML path.

    Returns:
        The constructed `folium.Map` object (also saved to disk as a
        side effect).
    """
    ensure_directories(save_path.parent)

    fmap = folium.Map(location=list(city_center), zoom_start=zoom_start, tiles="cartodbpositron")

    # Demand density heat layer
    heat_data = df[["latitude", "longitude", "demand_weight"]].values.tolist()
    HeatMap(heat_data, radius=12, blur=18, max_zoom=13, name="Commuter Demand Heat").add_to(fmap)

    # Recommended stop markers, sized/colored by total demand rank
    marker_layer = folium.FeatureGroup(name="Recommended Stops")
    max_demand = cluster_summary["total_demand"].max()

    for _, row in cluster_summary.iterrows():
        relative_demand = row["total_demand"] / max_demand if max_demand else 0
        radius = 8 + 12 * relative_demand
        color = "#dc2626" if row["rank_by_demand"] <= 3 else "#2563eb"

        popup_html = (
            f"<b>Recommended Stop #{int(row['rank_by_demand'])}</b><br>"
            f"Cluster ID: {int(row['cluster_id'])}<br>"
            f"Nearest area: {row['dominant_hotspot']}<br>"
            f"Commuters served: {int(row['n_commuters'])}<br>"
            f"Total demand: {row['total_demand']}<br>"
            f"Avg walk to stop: {row['avg_distance_to_stop_km']} km"
        )

        folium.CircleMarker(
            location=[row["stop_latitude"], row["stop_longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Stop #{int(row['rank_by_demand'])} ({row['dominant_hotspot']})",
        ).add_to(marker_layer)

    marker_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.save(str(save_path))
    logger.info("Saved interactive map to %s", save_path)
    return fmap
