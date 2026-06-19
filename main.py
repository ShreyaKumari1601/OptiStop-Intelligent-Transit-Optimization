"""
main.py
=======
OptiStop pipeline entry point.

Orchestrates the full workflow:
    1. Generate (or load cached) synthetic commuter demand data
    2. Quick exploratory summary (logged)
    3. Determine optimal k via Elbow Method + Silhouette Score
    4. Fit final weighted K-Means model -> recommended stop locations
    5. Compute cluster statistics & coverage analytics
    6. Render all plots + the interactive Folium map
    7. Export every artifact to outputs/

Usage:
    python main.py
    python main.py --n-commuters 6000 --force-regenerate
    python main.py --k 8                # skip auto-search, force k=8
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

# Ensure project root is importable when running `python main.py` directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analytics import (
    build_cluster_summary,
    coverage_analysis,
    rank_demand_hotspots,
    write_evaluation_report,
)
from src.clustering import TransitStopOptimizer
from src.config import CONFIG, Config
from src.data_loader import get_or_generate_demand_data
from src.utils.helpers import ensure_directories, setup_logger
from src.visualize import (
    build_interactive_map,
    plot_clusters,
    plot_demand_heatmap,
    plot_elbow_curve,
    plot_silhouette_scores,
)

logger = setup_logger(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline overrides."""
    parser = argparse.ArgumentParser(description="OptiStop transit stop optimization pipeline")
    parser.add_argument("--n-commuters", type=int, default=None, help="Number of synthetic commuters to simulate")
    parser.add_argument("--k", type=int, default=None, help="Force a specific number of clusters (skips auto-search)")
    parser.add_argument("--k-min", type=int, default=None, help="Minimum k to evaluate during auto-search")
    parser.add_argument("--k-max", type=int, default=None, help="Maximum k to evaluate during auto-search")
    parser.add_argument("--force-regenerate", action="store_true", help="Ignore cached data and regenerate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Apply CLI overrides on top of the default CONFIG (immutable dataclass)."""
    overrides = {}
    if args.n_commuters is not None:
        overrides["n_commuters"] = args.n_commuters
    if args.k_min is not None:
        overrides["k_min"] = args.k_min
    if args.k_max is not None:
        overrides["k_max"] = args.k_max
    if args.seed is not None:
        overrides["random_seed"] = args.seed

    return dataclasses.replace(CONFIG, **overrides) if overrides else CONFIG


def run_pipeline(config: Config, force_k: int | None = None, force_regenerate: bool = False) -> None:
    """Execute the end-to-end OptiStop pipeline.

    Args:
        config: Project configuration.
        force_k: If provided, skip the optimal-k search and fit directly
            with this many clusters.
        force_regenerate: If True, regenerate synthetic data even if a
            cached CSV exists.
    """
    start_time = time.perf_counter()
    logger.info("=" * 60)
    logger.info("OptiStop -- Intelligent Transit Stop Optimization")
    logger.info("City: %s | Config: %s", config.city_name, config.as_dict())
    logger.info("=" * 60)

    ensure_directories(config.data_dir, config.outputs_dir)

    # 1. Data ------------------------------------------------------------
    df = get_or_generate_demand_data(config, force_regenerate=force_regenerate)

    # 2. Quick EDA summary (logged, not exported as a separate file -- the
    #    full exploratory_analysis notebook covers richer EDA visuals) ----
    logger.info("EDA snapshot:\n%s", df[["latitude", "longitude", "demand_weight"]].describe())
    logger.info("Commuters per source hotspot:\n%s", df["source_hotspot"].value_counts())

    # 3. Optimal k search --------------------------------------------------
    optimizer = TransitStopOptimizer(config)

    if force_k is not None:
        logger.info("Skipping auto k-search; using forced k=%d", force_k)
        chosen_k = force_k
        # Still run a search so the elbow/silhouette plots have content.
        optimal_result = optimizer.find_optimal_k(df)
    else:
        optimal_result = optimizer.find_optimal_k(df)
        chosen_k = optimal_result.best_k

    # 4. Final fit -----------------------------------------------------
    labels, centers = optimizer.fit(df, k=chosen_k)
    overall_sil = optimizer.overall_silhouette(df)
    logger.info("Final model silhouette score (k=%d): %.4f", chosen_k, overall_sil)

    # 5. Analytics -------------------------------------------------------
    cluster_summary = build_cluster_summary(df, labels, centers)
    top_hotspots = rank_demand_hotspots(cluster_summary, top_n=5)
    coverage = coverage_analysis(df, centers, config.acceptable_walk_km)

    logger.info("Top demand hotspots:\n%s", top_hotspots.to_string(index=False))
    logger.info("Coverage analysis: %s", coverage)

    cluster_summary_path = config.outputs_dir / config.cluster_summary_filename
    cluster_summary.to_csv(cluster_summary_path, index=False)
    logger.info("Saved cluster summary to %s", cluster_summary_path)

    write_evaluation_report(
        path=config.outputs_dir / config.evaluation_metrics_filename,
        optimal_k_result=optimal_result,
        chosen_k=chosen_k,
        overall_silhouette=overall_sil,
        coverage=coverage,
        cluster_summary=cluster_summary,
        n_commuters=len(df),
    )

    # 6. Visualizations ----------------------------------------------------
    plot_elbow_curve(optimal_result, config.outputs_dir / config.elbow_plot_filename)
    plot_silhouette_scores(optimal_result, config.outputs_dir / config.silhouette_plot_filename)
    plot_clusters(df, labels, centers, config.outputs_dir / config.cluster_plot_filename)
    plot_demand_heatmap(df, config.outputs_dir / config.heatmap_plot_filename)
    build_interactive_map(
        df=df,
        cluster_summary=cluster_summary,
        city_center=config.city_center,
        zoom_start=config.map_default_zoom,
        save_path=config.outputs_dir / config.optimized_stops_map_filename,
    )

    elapsed = time.perf_counter() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.2f seconds. Outputs saved to: %s", elapsed, config.outputs_dir)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = build_config_from_args(args)

    try:
        run_pipeline(config, force_k=args.k, force_regenerate=args.force_regenerate)
    except Exception:
        logger.exception("Pipeline failed with an unhandled exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
