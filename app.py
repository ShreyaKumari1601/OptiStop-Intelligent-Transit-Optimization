"""
app.py
======
Streamlit dashboard for OptiStop.

Lets a user interactively adjust the number of synthetic commuters and
the clustering parameters, then view the resulting optimal bus stop
recommendations on a live interactive map alongside evaluation metrics.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.analytics import build_cluster_summary, coverage_analysis, rank_demand_hotspots
from src.clustering import TransitStopOptimizer
from src.config import CONFIG
from src.data_loader import CommuterDemandGenerator
from src.visualize import build_interactive_map, plot_clusters, plot_elbow_curve, plot_silhouette_scores

st.set_page_config(page_title="OptiStop | Transit Stop Optimizer", page_icon="🚌", layout="wide")


@st.cache_data(show_spinner=False)
def generate_data(n_commuters: int, seed: int) -> pd.DataFrame:
    """Generate (and cache) synthetic commuter demand data for given params."""
    config = dataclasses.replace(CONFIG, n_commuters=n_commuters, random_seed=seed)
    return CommuterDemandGenerator(config).generate()


def main() -> None:
    st.title("🚌 OptiStop: Intelligent Transit Stop Optimization")
    st.caption(
        "ML-driven bus stop placement using demand-weighted K-Means clustering "
        f"on synthetic commuter data for {CONFIG.city_name}."
    )

    with st.sidebar:
        st.header("⚙️ Parameters")
        n_commuters = st.slider("Number of synthetic commuters", 500, 8000, CONFIG.n_commuters, step=500)
        seed = st.number_input("Random seed", value=CONFIG.random_seed, step=1)

        st.divider()
        auto_k = st.checkbox("Auto-select optimal k (Elbow + Silhouette)", value=True)

        if auto_k:
            k_range = st.slider("k search range", 2, 20, (CONFIG.k_min, CONFIG.k_max))
        else:
            manual_k = st.slider("Number of stops (k)", 2, 20, 8)

        walk_km = st.slider("Acceptable walking distance (km)", 0.1, 1.5, CONFIG.acceptable_walk_km, step=0.05)

        run_button = st.button("🔄 Run Optimization", type="primary", use_container_width=True)

    if "has_run" not in st.session_state:
        st.session_state.has_run = False

    if run_button:
        st.session_state.has_run = True

    if not st.session_state.has_run:
        st.info("👈 Set your parameters in the sidebar and click **Run Optimization** to begin.")
        return

    with st.spinner("Generating commuter demand and optimizing stop locations..."):
        df = generate_data(n_commuters, seed)

        config = dataclasses.replace(
            CONFIG,
            n_commuters=n_commuters,
            random_seed=seed,
            k_min=k_range[0] if auto_k else 2,
            k_max=k_range[1] if auto_k else 20,
            acceptable_walk_km=walk_km,
        )

        optimizer = TransitStopOptimizer(config)
        optimal_result = optimizer.find_optimal_k(df)
        chosen_k = optimal_result.best_k if auto_k else manual_k

        labels, centers = optimizer.fit(df, k=chosen_k)
        overall_sil = optimizer.overall_silhouette(df)
        cluster_summary = build_cluster_summary(df, labels, centers)
        coverage = coverage_analysis(df, centers, walk_km)

    # ---- Top metrics row -------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Commuters simulated", f"{len(df):,}")
    m2.metric("Recommended stops (k)", chosen_k)
    m3.metric("Silhouette score", f"{overall_sil:.3f}")
    m4.metric("Within acceptable walk", f"{coverage['pct_within_acceptable_walk']:.1f}%")

    tab_map, tab_charts, tab_table, tab_about = st.tabs(
        ["🗺️ Interactive Map", "📊 Model Diagnostics", "📋 Cluster Details", "ℹ️ About"]
    )

    with tab_map:
        st.subheader("Recommended Stop Locations & Demand Heat")
        fmap = build_interactive_map(
            df=df,
            cluster_summary=cluster_summary,
            city_center=config.city_center,
            zoom_start=config.map_default_zoom,
            save_path=config.outputs_dir / "optimized_stops_map.html",
        )
        st_folium(fmap, width=None, height=560, returned_objects=[])

    with tab_charts:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Elbow Method")
            elbow_path = config.outputs_dir / "elbow_curve_app.png"
            plot_elbow_curve(optimal_result, elbow_path)
            st.image(str(elbow_path))
        with col2:
            st.subheader("Silhouette Scores")
            sil_path = config.outputs_dir / "silhouette_scores_app.png"
            plot_silhouette_scores(optimal_result, sil_path)
            st.image(str(sil_path))

        st.subheader("Cluster Map (static)")
        cluster_plot_path = config.outputs_dir / "cluster_plot_app.png"
        plot_clusters(df, labels, centers, cluster_plot_path)
        st.image(str(cluster_plot_path))

    with tab_table:
        st.subheader("Per-Stop Statistics")
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        st.subheader("Top 5 Highest-Demand Stops")
        st.dataframe(
            rank_demand_hotspots(cluster_summary, top_n=5), use_container_width=True, hide_index=True
        )

        csv_bytes = cluster_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download cluster_summary.csv", csv_bytes, file_name="cluster_summary.csv", mime="text/csv"
        )

    with tab_about:
        st.markdown(
            """
            **OptiStop** identifies optimal bus stop locations by clustering
            synthetic commuter demand points with a demand-weighted K-Means
            model. The optimal number of stops (k) is selected automatically
            using the **Elbow Method** (inertia) combined with the
            **Silhouette Score**.

            - Adjust commuter volume and the k search range from the sidebar.
            - Cluster centroids (weighted by simulated demand) become the
              recommended stop coordinates.
            - The coverage metric estimates what share of commuters fall
              within an acceptable walking distance of their nearest
              recommended stop.

            Built with scikit-learn, Folium, and Streamlit.
            """
        )


if __name__ == "__main__":
    main()
