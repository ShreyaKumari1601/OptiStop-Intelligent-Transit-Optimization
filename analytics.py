"""
analytics.py
=============
Post-clustering analytics: per-cluster statistics, demand hotspot ranking,
walking-distance coverage analysis, and evaluation report generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.clustering import OptimalKResult
from src.utils.helpers import ensure_directories, nearest_distance_km

logger = logging.getLogger("optistop")


def build_cluster_summary(
    df: pd.DataFrame, labels: np.ndarray, centers: np.ndarray
) -> pd.DataFrame:
    """Compute per-cluster (recommended stop) statistics.

    Args:
        df: Original demand DataFrame with `latitude`, `longitude`,
            `demand_weight`.
        labels: Cluster label assigned to each row of `df`.
        centers: Array of shape (k, 2) with [latitude, longitude] of each
            cluster centroid (the recommended stop location).

    Returns:
        DataFrame, one row per cluster, sorted by total demand descending,
        with columns: cluster_id, stop_latitude, stop_longitude,
        n_commuters, total_demand, avg_demand, dominant_hotspot,
        avg_distance_to_stop_km.
    """
    work_df = df.copy()
    work_df["cluster"] = labels

    rows = []
    for cluster_id, group in work_df.groupby("cluster"):
        center_lat, center_lon = centers[cluster_id]
        dist_km = nearest_distance_km(
            group["latitude"].to_numpy(),
            group["longitude"].to_numpy(),
            [center_lat],
            [center_lon],
        )
        dominant_hotspot = (
            group["source_hotspot"].value_counts().idxmax()
            if "source_hotspot" in group.columns
            else "n/a"
        )
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "stop_latitude": round(float(center_lat), 6),
                "stop_longitude": round(float(center_lon), 6),
                "n_commuters": int(len(group)),
                "total_demand": round(float(group["demand_weight"].sum()), 2),
                "avg_demand": round(float(group["demand_weight"].mean()), 3),
                "dominant_hotspot": dominant_hotspot,
                "avg_distance_to_stop_km": round(float(dist_km.mean()), 3),
                "max_distance_to_stop_km": round(float(dist_km.max()), 3),
            }
        )

    summary = pd.DataFrame(rows).sort_values("total_demand", ascending=False)
    summary = summary.reset_index(drop=True)
    summary.insert(0, "rank_by_demand", summary.index + 1)
    return summary


def rank_demand_hotspots(cluster_summary: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the top-N highest-demand clusters as priority deployment sites.

    Args:
        cluster_summary: Output of `build_cluster_summary`.
        top_n: Number of top hotspots to return.

    Returns:
        The top-N rows of `cluster_summary`, ranked by total demand.
    """
    return cluster_summary.head(top_n).reset_index(drop=True)


def coverage_analysis(
    df: pd.DataFrame, centers: np.ndarray, acceptable_walk_km: float
) -> Dict[str, float]:
    """Estimate what share of commuters fall within an acceptable walk of a stop.

    Args:
        df: Demand DataFrame with `latitude`, `longitude`.
        centers: Array of shape (k, 2) of recommended stop coordinates.
        acceptable_walk_km: Distance (km) considered an acceptable walk.

    Returns:
        Dict with `mean_distance_km`, `median_distance_km`,
        `pct_within_acceptable_walk`, and `acceptable_walk_km`.
    """
    distances = nearest_distance_km(
        df["latitude"].to_numpy(),
        df["longitude"].to_numpy(),
        centers[:, 0],
        centers[:, 1],
    )
    pct_within = float(np.mean(distances <= acceptable_walk_km) * 100)

    return {
        "mean_distance_km": round(float(np.mean(distances)), 4),
        "median_distance_km": round(float(np.median(distances)), 4),
        "pct_within_acceptable_walk": round(pct_within, 2),
        "acceptable_walk_km": acceptable_walk_km,
    }


def write_evaluation_report(
    path: Path,
    optimal_k_result: OptimalKResult,
    chosen_k: int,
    overall_silhouette: float,
    coverage: Dict[str, float],
    cluster_summary: pd.DataFrame,
    n_commuters: int,
) -> None:
    """Write a human-readable evaluation report to a text file.

    Args:
        path: Destination path for the report.
        optimal_k_result: Results of the k-search (elbow + silhouette scan).
        chosen_k: The k value used for the final model.
        overall_silhouette: Silhouette score of the final fitted model.
        coverage: Output of `coverage_analysis`.
        cluster_summary: Output of `build_cluster_summary`.
        n_commuters: Total number of synthetic commuters simulated.
    """
    ensure_directories(path.parent)

    lines = []
    lines.append("=" * 60)
    lines.append("OptiStop -- Evaluation Metrics Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total simulated commuters analyzed : {n_commuters}")
    lines.append(f"Candidate k range evaluated         : {optimal_k_result.k_values[0]}–{optimal_k_result.k_values[-1]}")
    lines.append(f"Chosen optimal k (recommended stops): {chosen_k}")
    lines.append(f"Final model silhouette score        : {overall_silhouette:.4f}")
    lines.append("")
    lines.append("-- Elbow Method / Silhouette scan --")
    lines.append(f"{'k':>4} | {'inertia':>14} | {'silhouette':>10}")
    for k, inertia, sil in zip(
        optimal_k_result.k_values, optimal_k_result.inertias, optimal_k_result.silhouette_scores
    ):
        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "   n/a"
        lines.append(f"{k:>4} | {inertia:>14.2f} | {sil_str:>10}")
    lines.append("")
    lines.append("-- Coverage Analysis --")
    lines.append(f"Acceptable walk distance     : {coverage['acceptable_walk_km']} km")
    lines.append(f"Mean distance to nearest stop: {coverage['mean_distance_km']} km")
    lines.append(f"Median distance to nearest stop: {coverage['median_distance_km']} km")
    lines.append(
        f"Commuters within acceptable walk: {coverage['pct_within_acceptable_walk']}%"
    )
    lines.append("")
    lines.append("-- Top 5 Highest-Demand Recommended Stops --")
    top5 = rank_demand_hotspots(cluster_summary, top_n=5)
    for _, row in top5.iterrows():
        lines.append(
            f"  #{int(row['rank_by_demand'])}: cluster {int(row['cluster_id'])} "
            f"near '{row['dominant_hotspot']}' "
            f"-> ({row['stop_latitude']}, {row['stop_longitude']}) "
            f"| total_demand={row['total_demand']} | commuters={int(row['n_commuters'])}"
        )
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    path.write_text(report_text, encoding="utf-8")
    logger.info("Evaluation report written to %s", path)
