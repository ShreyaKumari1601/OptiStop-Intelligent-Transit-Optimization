"""
clustering.py
==============
K-Means based geospatial clustering for optimal transit stop placement.

This module determines the best number of clusters (k) using the Elbow
Method (inertia) combined with Silhouette Score, then fits a final
K-Means model whose cluster centroids become the recommended bus stop
locations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import Config

logger = logging.getLogger("optistop")


@dataclass
class OptimalKResult:
    """Results of the optimal-k search across a range of cluster counts.

    Attributes:
        k_values: The candidate cluster counts that were evaluated.
        inertias: Within-cluster sum of squares for each k (Elbow Method).
        silhouette_scores: Silhouette score for each k (higher is better,
            range [-1, 1]).
        best_k: The recommended k, chosen as the one maximizing
            silhouette score among the evaluated candidates.
    """

    k_values: List[int]
    inertias: List[float]
    silhouette_scores: List[float]
    best_k: int


class TransitStopOptimizer:
    """Finds optimal bus stop locations via weighted K-Means clustering.

    Commuter `demand_weight` is used as the K-Means `sample_weight`, so
    high-demand commuters pull cluster centroids toward themselves more
    strongly than low-demand ones -- this is what makes the resulting
    centroids genuine "demand-weighted" stop recommendations rather than
    plain geographic midpoints.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the optimizer.

        Args:
            config: Project configuration with clustering hyperparameters
                (k search range, KMeans init/iteration settings, seed).
        """
        self.config = config
        self.model: KMeans | None = None
        self.labels_: np.ndarray | None = None
        self.optimal_k_result: OptimalKResult | None = None

    @staticmethod
    def _validate_inputs(df: pd.DataFrame) -> None:
        """Ensure the input DataFrame has the required columns and rows."""
        required_cols = {"latitude", "longitude", "demand_weight"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")
        if df.empty:
            raise ValueError("Input demand DataFrame is empty; cannot cluster.")
        if df[["latitude", "longitude"]].isnull().any().any():
            raise ValueError("Input data contains null latitude/longitude values.")

    def find_optimal_k(
        self, df: pd.DataFrame, sample_size: int | None = 1500
    ) -> OptimalKResult:
        """Search the configured k range for the best cluster count.

        Uses the Elbow Method (inertia) for visualization and Silhouette
        Score as the quantitative tie-breaker for selecting `best_k`.

        Args:
            df: Demand DataFrame with `latitude`, `longitude`, `demand_weight`.
            sample_size: To keep silhouette computation fast on larger
                datasets, scores are computed on a random subsample of at
                most this many points. Set to None to use the full dataset.

        Returns:
            An `OptimalKResult` summarizing inertia and silhouette scores
            per k, plus the recommended `best_k`.
        """
        self._validate_inputs(df)
        X = df[["latitude", "longitude"]].to_numpy()
        weights = df["demand_weight"].to_numpy()

        k_values = list(range(self.config.k_min, self.config.k_max + 1))
        inertias: List[float] = []
        sil_scores: List[float] = []

        if sample_size is not None and len(X) > sample_size:
            rng = np.random.default_rng(self.config.random_seed)
            sample_idx = rng.choice(len(X), size=sample_size, replace=False)
        else:
            sample_idx = np.arange(len(X))

        logger.info("Searching optimal k in range [%d, %d]", k_values[0], k_values[-1])

        for k in k_values:
            kmeans = KMeans(
                n_clusters=k,
                n_init=self.config.kmeans_n_init,
                max_iter=self.config.kmeans_max_iter,
                random_state=self.config.random_seed,
            )
            labels = kmeans.fit_predict(X, sample_weight=weights)
            inertias.append(float(kmeans.inertia_))

            if k > 1:
                score = silhouette_score(X[sample_idx], labels[sample_idx])
            else:
                score = float("nan")
            sil_scores.append(float(score))

            logger.debug("k=%d | inertia=%.2f | silhouette=%.4f", k, kmeans.inertia_, score)

        best_k = self._select_best_k(k_values, sil_scores)
        logger.info("Optimal k selected: %d", best_k)

        self.optimal_k_result = OptimalKResult(
            k_values=k_values, inertias=inertias, silhouette_scores=sil_scores, best_k=best_k
        )
        return self.optimal_k_result

    @staticmethod
    def _select_best_k(
        k_values: List[int], sil_scores: List[float], tolerance: float = 0.02
    ) -> int:
        """Pick the smallest k whose silhouette score is within `tolerance`
        of the best observed score.

        A pure argmax over silhouette scores tends to favor the largest k
        in the search range whenever scores plateau (a common pattern once
        k exceeds the data's natural number of groups), which produces an
        "optimal" k that looks disconnected from the visual elbow. This
        parsimony rule keeps the simplest model that is statistically
        indistinguishable from the best one -- standard practice when
        silhouette curves are relatively flat across a range of k.

        Args:
            k_values: Candidate k values evaluated.
            sil_scores: Corresponding silhouette scores (may contain NaN
                for k=1, which is skipped).
            tolerance: Absolute silhouette-score tolerance defining
                "close enough" to the best score.

        Returns:
            The selected k.
        """
        valid = [(k, s) for k, s in zip(k_values, sil_scores) if not np.isnan(s)]
        if not valid:
            return k_values[0]

        best_score = max(s for _, s in valid)
        candidates = [k for k, s in valid if s >= best_score - tolerance]
        return min(candidates)

    def fit(self, df: pd.DataFrame, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the final demand-weighted K-Means model.

        Args:
            df: Demand DataFrame with `latitude`, `longitude`, `demand_weight`.
            k: Number of clusters (recommended stops) to fit.

        Returns:
            Tuple of (cluster_labels, cluster_centers) where centers is an
            array of shape (k, 2) of [latitude, longitude] pairs.
        """
        self._validate_inputs(df)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        X = df[["latitude", "longitude"]].to_numpy()
        weights = df["demand_weight"].to_numpy()

        self.model = KMeans(
            n_clusters=k,
            n_init=self.config.kmeans_n_init,
            max_iter=self.config.kmeans_max_iter,
            random_state=self.config.random_seed,
        )
        self.labels_ = self.model.fit_predict(X, sample_weight=weights)

        logger.info("Fitted final KMeans model with k=%d clusters", k)
        return self.labels_, self.model.cluster_centers_

    def overall_silhouette(self, df: pd.DataFrame) -> float:
        """Compute the silhouette score for the currently fitted model.

        Args:
            df: The same demand DataFrame used to fit the model.

        Returns:
            Silhouette score (float in [-1, 1]).

        Raises:
            RuntimeError: If `fit()` has not been called yet.
        """
        if self.model is None or self.labels_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = df[["latitude", "longitude"]].to_numpy()
        return float(silhouette_score(X, self.labels_))
