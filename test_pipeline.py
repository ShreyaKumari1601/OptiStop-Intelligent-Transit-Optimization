"""
tests/test_pipeline.py
=======================
Unit tests covering the core OptiStop pipeline components:
data generation, clustering, analytics, and geo-utility helpers.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.analytics import build_cluster_summary, coverage_analysis, rank_demand_hotspots
from src.clustering import TransitStopOptimizer
from src.config import CONFIG, Config, HotspotConfig
from src.data_loader import CommuterDemandGenerator
from src.utils.helpers import haversine_km, km_to_deg_lat, km_to_deg_lon, nearest_distance_km


@pytest.fixture()
def small_config() -> Config:
    """A small, fast-to-run config for unit tests."""
    return dataclasses.replace(
        CONFIG,
        n_commuters=300,
        k_min=2,
        k_max=6,
        random_seed=7,
    )


@pytest.fixture()
def demand_df(small_config: Config) -> pd.DataFrame:
    """A small synthetic demand dataset for reuse across tests."""
    return CommuterDemandGenerator(small_config).generate()


class TestHelpers:
    def test_haversine_known_distance(self) -> None:
        """Distance between two well-known points should be approximately correct."""
        # MG Road to Whitefield, Bangalore: ~15-17 km in a straight line.
        dist = haversine_km(12.9758, 77.6045, 12.9698, 77.7500)
        assert 14.0 < dist < 18.0

    def test_haversine_zero_distance(self) -> None:
        """Distance from a point to itself must be zero."""
        assert haversine_km(12.97, 77.59, 12.97, 77.59) == pytest.approx(0.0, abs=1e-9)

    def test_km_to_deg_lat_roundtrip(self) -> None:
        """1 km should be a small positive fraction of a degree of latitude."""
        deg = km_to_deg_lat(1.0)
        assert 0.0 < deg < 0.02

    def test_km_to_deg_lon_varies_with_latitude(self) -> None:
        """Longitude degree-size should shrink at higher latitudes."""
        deg_at_equator = km_to_deg_lon(1.0, at_lat=0.0)
        deg_at_60 = km_to_deg_lon(1.0, at_lat=60.0)
        assert deg_at_60 > deg_at_equator

    def test_nearest_distance_km_picks_closest(self) -> None:
        """Each point should be matched to its truly nearest target."""
        points_lat = np.array([12.97, 12.85])
        points_lon = np.array([77.60, 77.66])
        targets_lat = [12.97, 12.85]
        targets_lon = [77.60, 77.66]
        dists = nearest_distance_km(points_lat, points_lon, targets_lat, targets_lon)
        np.testing.assert_allclose(dists, [0.0, 0.0], atol=1e-6)


class TestDataLoader:
    def test_generate_returns_correct_row_count(self, small_config: Config) -> None:
        df = CommuterDemandGenerator(small_config).generate()
        # Allow small rounding slack from per-hotspot weight*n calculations.
        assert abs(len(df) - small_config.n_commuters) <= len(small_config.hotspots)

    def test_generate_required_columns(self, demand_df: pd.DataFrame) -> None:
        for col in ["latitude", "longitude", "demand_weight", "source_hotspot"]:
            assert col in demand_df.columns

    def test_generate_no_nulls(self, demand_df: pd.DataFrame) -> None:
        assert not demand_df[["latitude", "longitude", "demand_weight"]].isnull().any().any()

    def test_demand_weight_is_positive(self, demand_df: pd.DataFrame) -> None:
        assert (demand_df["demand_weight"] > 0).all()

    def test_invalid_hotspot_weights_raise(self) -> None:
        bad_config = dataclasses.replace(
            CONFIG,
            hotspots=[HotspotConfig("A", 12.9, 77.6, 0.5, 1.0)],  # sums to 0.5, not 1.0
        )
        with pytest.raises(ValueError):
            CommuterDemandGenerator(bad_config)

    def test_zero_commuters_raises(self) -> None:
        bad_config = dataclasses.replace(CONFIG, n_commuters=0)
        with pytest.raises(ValueError):
            CommuterDemandGenerator(bad_config)

    def test_reproducible_with_same_seed(self, small_config: Config) -> None:
        df1 = CommuterDemandGenerator(small_config).generate()
        df2 = CommuterDemandGenerator(small_config).generate()
        pd.testing.assert_frame_equal(df1, df2)


class TestClustering:
    def test_find_optimal_k_returns_valid_result(
        self, small_config: Config, demand_df: pd.DataFrame
    ) -> None:
        optimizer = TransitStopOptimizer(small_config)
        result = optimizer.find_optimal_k(demand_df, sample_size=None)
        assert small_config.k_min <= result.best_k <= small_config.k_max
        assert len(result.inertias) == len(result.k_values)
        assert len(result.silhouette_scores) == len(result.k_values)

    def test_fit_produces_expected_cluster_count(
        self, small_config: Config, demand_df: pd.DataFrame
    ) -> None:
        optimizer = TransitStopOptimizer(small_config)
        labels, centers = optimizer.fit(demand_df, k=4)
        assert centers.shape == (4, 2)
        assert len(np.unique(labels)) == 4
        assert len(labels) == len(demand_df)

    def test_fit_rejects_invalid_k(self, small_config: Config, demand_df: pd.DataFrame) -> None:
        optimizer = TransitStopOptimizer(small_config)
        with pytest.raises(ValueError):
            optimizer.fit(demand_df, k=0)

    def test_overall_silhouette_requires_fit_first(
        self, small_config: Config, demand_df: pd.DataFrame
    ) -> None:
        optimizer = TransitStopOptimizer(small_config)
        with pytest.raises(RuntimeError):
            optimizer.overall_silhouette(demand_df)

    def test_missing_columns_raise(self, small_config: Config) -> None:
        optimizer = TransitStopOptimizer(small_config)
        bad_df = pd.DataFrame({"lat": [1, 2], "lon": [3, 4]})
        with pytest.raises(ValueError):
            optimizer.fit(bad_df, k=2)

    def test_empty_dataframe_raises(self, small_config: Config) -> None:
        optimizer = TransitStopOptimizer(small_config)
        empty_df = pd.DataFrame(columns=["latitude", "longitude", "demand_weight"])
        with pytest.raises(ValueError):
            optimizer.fit(empty_df, k=2)


class TestAnalytics:
    def test_cluster_summary_row_count_matches_k(
        self, small_config: Config, demand_df: pd.DataFrame
    ) -> None:
        optimizer = TransitStopOptimizer(small_config)
        labels, centers = optimizer.fit(demand_df, k=4)
        summary = build_cluster_summary(demand_df, labels, centers)
        assert len(summary) == 4

    def test_cluster_summary_demand_sums_to_total(
        self, small_config: Config, demand_df: pd.DataFrame
    ) -> None:
        optimizer = TransitStopOptimizer(small_config)
        labels, centers = optimizer.fit(demand_df, k=4)
        summary = build_cluster_summary(demand_df, labels, centers)
        assert summary["total_demand"].sum() == pytest.approx(
            demand_df["demand_weight"].sum(), rel=1e-3
        )

    def test_rank_demand_hotspots_returns_sorted_top_n(
        self, small_config: Config, demand_df: pd.DataFrame
    ) -> None:
        optimizer = TransitStopOptimizer(small_config)
        labels, centers = optimizer.fit(demand_df, k=4)
        summary = build_cluster_summary(demand_df, labels, centers)
        top2 = rank_demand_hotspots(summary, top_n=2)
        assert len(top2) == 2
        assert top2["total_demand"].is_monotonic_decreasing

    def test_coverage_analysis_percentage_in_valid_range(
        self, small_config: Config, demand_df: pd.DataFrame
    ) -> None:
        optimizer = TransitStopOptimizer(small_config)
        _, centers = optimizer.fit(demand_df, k=4)
        coverage = coverage_analysis(demand_df, centers, acceptable_walk_km=0.5)
        assert 0.0 <= coverage["pct_within_acceptable_walk"] <= 100.0
        assert coverage["mean_distance_km"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
