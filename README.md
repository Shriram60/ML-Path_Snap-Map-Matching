# PathSnap: Map-Matching and Route Reconstruction

PathSnap matches noisy GPS points to road segments and rebuilds the route that was driven. For each GPS point it finds nearby candidate roads, a trained Random Forest scores each candidate, and an HMM with Viterbi decoding picks one connected route.

The system runs on two laptops. Laptop 1 runs the API, model and decoder. Laptop 2 runs the evaluator UI in a browser.

## Repository layout

| Folder | Contents |
|---|---|
| `data_processing/` | Reads trip files, cleans tracks, projects coordinates |
| `features/` | Reference labels, candidate generation, feature extraction, candidate tuning |
| `eda/` | Dataset inventory, GPS quality, road network and route-fit analysis, Folium maps |
| `models/` | Model training and comparison, inference, saved model in `models/artifacts/` |
| `tuning/` | Bayesian hyperparameter optimisation with Optuna |
| `evaluation/` | Classifier metrics |
| `results/` | Model comparison, tuning results and plots |
| `hmm/` | Viterbi decoder, decoder tuning, route metrics, `hmm_config.json` |
| `api/` | FastAPI backend |
| `ui/` | Evaluator interface |
| `visualisations/` | Folium maps of predicted routes |
| `deployment/` | Requirements, start scripts and the full two-laptop guide |
| `qa/`, `tests/` | API tests and pipeline tests |
| `team/` | Work split and team updates |

## Installation

Use Python 3.12 and run every command from the repo root.

```
git clone https://github.com/Shriram60/ML-Path_Snap-Map-Matching.git
cd ML-Path_Snap-Map-Matching
python -m venv .venv
.venv\Scripts\activate
pip install -r deployment\requirements-api.txt
pip install matplotlib pytest
```

The requirements file pins the scikit-learn version the saved model was created with.

The dataset is not in this repo. Set these two folders before running anything:

```
set PATHSNAP_DATA=C:\path\to\folder\with\train_and_student_test
set PATHSNAP_WORK=C:\path\to\work\folder
```

`PATHSNAP_DATA` holds `train/` and `student_test/`. Generated tables are written to `PATHSNAP_WORK`.

## Running the demo

**Laptop 1, API**

```
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator\trips
deployment\run_api.bat
```

Without `PATHSNAP_EVAL_DIR`, the API serves `student_test`. Keep this window visible, because it logs each request and the model files in use. Run `ipconfig` to find the laptop's IPv4 address.

**Laptop 2, UI**

```
deployment\run_ui.bat <laptop-1-ip>
```

Then open `http://localhost:5500/index.html?api=http://<laptop-1-ip>:8000`.

Both laptops must be on the same network. A phone hotspot is the safest choice. Firewall steps, the demo checklist and troubleshooting are in [`deployment/README.md`](deployment/README.md).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` |
| `GET /trips` | Lists trip IDs in the evaluation folder |
| `POST /predict` | Takes `{"trip_id": "<id>"}` and returns GPS points, predicted arcs, the route and metrics |

The reference route is only used on the server, after prediction, to compute metrics. It never reaches the model or the UI.

## Reproduce the results

Run these in order. Each step needs the ones before it.

```
python -m features.candidate_tuning
python -m features.build_dataset
python -m models.train_compare
python -m tuning.bayes_opt --model random_forest --timeout 900 --search-frac 0.05
python -m hmm.final_scores
python -m hmm.evaluate --scores "%PATHSNAP_WORK%\final_scores" --source "final tuned random forest"
```

| Step | Output |
|---|---|
| Candidate tuning | `features/outputs/candidate_tuning.csv` |
| Dataset build | Candidate and reference tables in `PATHSNAP_WORK`, summaries in `features/outputs/` |
| Model comparison | `results/model_comparison.csv`, `results/per_model/`, `results/plots/` |
| Bayesian tuning | `results/tuning/`, `models/artifacts/` |
| HMM tuning and comparison | `hmm/hmm_config.json`, `hmm/outputs/` |

All hyperparameter tuning uses Bayesian optimisation with Optuna. Evaluator trips are not used for training, feature work, model choice or tuning.

`tuning.bayes_opt` stops at a time limit, so its trial count can differ between machines.

## Exploration and maps

```
python eda/inventory.py
python eda/gps_quality.py
python eda/road_network.py
python eda/route_fit.py
python eda/trip_map.py --selected
python -m visualisations.route_map --all --partition student_test
```

## Tests

```
python -m pytest tests
```

With the API running on port 8000 and serving `student_test`:

```
python -m unittest qa.test_api -v
```
