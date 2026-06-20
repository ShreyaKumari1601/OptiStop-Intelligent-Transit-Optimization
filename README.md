# 🚌 OptiStop: Intelligent Transit Stop Optimization System

> ML-driven geospatial clustering to recommend optimal bus stop locations from commuter demand density — built and validated on a synthetic Bangalore dataset.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen.svg)](tests/test_pipeline.py)

---

## 📌 Problem

Urban public transport systems often place bus stops based on historical convention or road geometry rather than actual commuter demand. This causes:

- Long, uneven walking distances to the nearest stop
- Over-served low-demand corridors and under-served high-demand pockets
- No data-driven way to prioritize where new stops should go first

## 🎯 Objective

Use **machine learning–based geospatial clustering** to recommend bus stop locations that minimize commuter walking distance while reflecting real demand density — and to **quantify** how good (or bad) current coverage would be under those recommendations.

## 🧠 Approach

1. **Simulate realistic commuter demand** as a mixture of Gaussian clusters around ten well-known Bangalore demand zones (Koramangala, Whitefield, Electronic City, etc.), plus background noise.
2. **Explore** the spatial and statistical structure of demand (EDA notebook).
3. **Cluster** commuters with **demand-weighted K-Means** — each commuter's simulated trip volume acts as a `sample_weight`, so centroids are pulled toward high-demand areas, not just geographic midpoints.
4. **Select k automatically** using the **Elbow Method** (inertia) cross-checked with the **Silhouette Score**, applying a parsimony rule that picks the smallest k statistically close to the best score (avoids overfitting to noisy local maxima at the edge of the search range).
5. **Recommend** each cluster centroid as an optimal stop, and **report** coverage: what share of commuters fall within an acceptable walking distance of their nearest recommended stop.
6. **Visualize** everything statically (Matplotlib/Seaborn) and interactively (Folium + Streamlit).

---

## 🏗️ Architecture

```
                ┌─────────────────────┐
                │   src/config.py     │   ← single source of truth
                │  (hotspots, paths,  │     for every tunable param
                │   k-range, seed)    │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  src/data_loader.py │   Synthetic commuter demand
                │ CommuterDemand      │   (Gaussian mixture + noise)
                │   Generator         │
                └──────────┬──────────┘
                           │ DataFrame[lat, lon, demand_weight]
                ┌──────────▼──────────┐
                │  src/clustering.py  │   Elbow + Silhouette search
                │ TransitStop         │   → weighted K-Means fit
                │   Optimizer         │
                └──────────┬──────────┘
                           │ labels, cluster_centers
                ┌──────────▼──────────┐
                │  src/analytics.py   │   Cluster stats, hotspot ranking,
                │                     │   coverage %, evaluation report
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  src/visualize.py   │   Elbow/Silhouette/Cluster plots,
                │                     │   KDE heatmap, Folium map
                └──────────┬──────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
            outputs/*.png       outputs/optimized_stops_map.html
            outputs/*.csv       outputs/evaluation_metrics.txt
                 │
                 ▼
         app.py (Streamlit dashboard — same pipeline, interactive)
```

**Design principles:** config-driven (no magic numbers scattered in code), every module independently testable, logging instead of `print`, type hints + docstrings throughout, and a clean separation between *data*, *modeling*, *analytics*, and *presentation* layers.

---

## 📂 Project Structure

```
OptiStop/
├── main.py                       # Pipeline entry point (CLI)
├── app.py                        # Streamlit interactive dashboard
├── requirements.txt              # Full dependency set
├── requirements-deploy.txt       # Lean set for Streamlit Cloud
├── README.md
├── LICENSE
├── .gitignore
│
├── data/                         # Generated/cached synthetic dataset
├── outputs/                      # All exported artifacts (gitignored)
│
├── src/
│   ├── config.py                 # Dataclass-based central config
│   ├── data_loader.py            # Synthetic demand generation
│   ├── clustering.py             # K-Means + optimal-k search
│   ├── analytics.py              # Cluster stats & coverage analysis
│   ├── visualize.py              # All static + interactive plots
│   └── utils/
│       └── helpers.py            # Logging, haversine, geo conversions
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── tests/
│   └── test_pipeline.py          # 22 unit tests, pytest
│
└── screenshots/                  # Add your own screenshots here
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/OptiStop.git
cd OptiStop

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Requires **Python 3.11+**.

---

## 🚀 Usage

### Run the full pipeline (CLI)

```bash
python main.py
```

This will:
1. Generate (or load cached) synthetic commuter demand for Bangalore
2. Search k ∈ [2, 12] via Elbow Method + Silhouette Score
3. Fit the final weighted K-Means model
4. Export every artifact to `outputs/`

**Useful flags:**

```bash
python main.py --n-commuters 6000        # simulate more commuters
python main.py --k 8                     # force exactly 8 stops (skip auto-search)
python main.py --k-min 3 --k-max 10      # narrow the auto-search range
python main.py --force-regenerate        # ignore cached data/*.csv
python main.py --seed 123                # different random scenario
```

### Run the interactive dashboard

```bash
streamlit run app.py
```

Adjust commuter volume, the k search range, and acceptable walking distance live; view the Folium map, diagnostic charts, and per-stop statistics in-browser.

### Explore the data interactively

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Run the test suite

```bash
pytest tests/ -v
```

---

## 📊 Results

Running the default pipeline (4,000 simulated commuters, seed=42) produces:

| Metric | Value |
|---|---|
| Optimal k (recommended stops) | **10** |
| Final silhouette score | **0.588** |
| Mean distance to nearest recommended stop | **1.62 km** |
| Commuters within 0.4 km of a stop | **7.4%** |

> The low "within acceptable walk" figure is the point: it quantifies *current-style* single-stop-per-zone coverage and exposes exactly where a real deployment would need **multiple** stops per high-demand cluster — see [Future Improvements](#-future-improvements).

**Top 5 demand hotspots identified:**

| Rank | Area | Commuters | Total Demand |
|---|---|---|---|
| 1 | Koramangala | 747 | 833.18 |
| 2 | Whitefield | 514 | 580.59 |
| 3 | Electronic City | 484 | 535.47 |
| 4 | MG Road / Trinity | 434 | 488.70 |
| 5 | Indiranagar | 395 | 447.55 |

All numbers are reproducible — re-run `python main.py` and check `outputs/evaluation_metrics.txt`.

### Exported artifacts

- `outputs/optimized_stops_map.html` — interactive Folium map (demand heatmap + clickable stop markers)
- `outputs/cluster_summary.csv` — per-stop statistics
- `outputs/evaluation_metrics.txt` — full elbow/silhouette scan + coverage report
- `outputs/elbow_curve.png`, `silhouette_scores.png`, `cluster_plot.png`, `demand_heatmap.png`

### Screenshots

> Add your own screenshots to `screenshots/` and reference them here, e.g.:
>
> ```markdown
> ![Interactive Map](screenshots/map_view.png)
> ![Cluster Plot](screenshots/cluster_plot.png)
> ![Streamlit Dashboard](screenshots/dashboard.png)
> ```

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public or private with Streamlit Cloud access).
2. Go to [share.streamlit.io](https://share.streamlit.io/) → **New app**.
3. Select your repo, branch, and set **Main file path** to `app.py`.
4. Under **Advanced settings**, set the requirements file to `requirements-deploy.txt` (leaner/faster build), or leave default `requirements.txt`.
5. Deploy. First build takes a few minutes; subsequent pushes auto-redeploy.

No secrets or API keys are required — the entire app runs on synthetic, locally generated data.

---

## 🔮 Future Improvements

- **Multi-stop allocation per zone**: extend beyond one centroid per cluster to place several stops within large/high-density clusters (e.g. via recursive sub-clustering or capacitated facility location).
- **Real demand data integration**: replace synthetic generation with actual smart-card/GTFS ridership data when available.
- **Road-network-aware distance**: replace haversine "as-the-crow-flies" distance with actual walking-network routing (e.g. OSMnx) for realistic coverage metrics.
- **Alternative algorithms**: compare K-Means against DBSCAN/HDBSCAN (density-based, handles irregular cluster shapes) and weighted facility-location optimization.
- **Multi-objective optimization**: balance stop count (cost) against coverage (commuter benefit) using a Pareto frontier instead of a single k.
- **Time-of-day demand modeling**: simulate peak vs. off-peak demand shifts and recommend dynamic/flexible stop strategies.

---

## 🛠️ Tech Stack

`Python 3.11` · `pandas` · `NumPy` · `scikit-learn` · `Folium` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
