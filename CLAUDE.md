# PathSnap team instructions

PathSnap is our UE24CS352A hackathon project: match noisy GPS points to road segments, then reconstruct the route with an HMM and Viterbi. Four people work on this. Shriram, Sharanya and Shriya each have a Claude on their own laptop. Abdullah codes without Claude, so make requests to him specific and self-contained.

Ask your user who they are if it is not clear, then read their brief:

| Person | Brief | Owns |
|---|---|---|
| Shriram | `team/SHRIRAM.md` | `data_processing/`, `features/`, `hmm/` |
| Sharanya | `team/SHARANYA.md` | `models/`, `tuning/`, `evaluation/`, `results/` |
| Shriya | `team/SHRIYA.md` | `api/`, `ui/`, `visualisations/`, `deployment/` |
| Abdullah | `team/ABDULLAH.md` | `eda/`, `qa/`, `report/` |

The hackathon PDFs are the source of truth. If this file disagrees with them, follow the PDFs and post the mismatch in your update file.

## Evaluation rubric

Total 25 marks, 5 for each:

| Component | Main owner |
|---|---|
| Data preprocessing and feature extraction | Shriram and Abdullah |
| Model selection and training | Sharanya |
| Hyperparameter tuning | Sharanya |
| ML Ops | Shriya |
| Demo and deliverables | Shriya and Abdullah |

Evaluators are told to check the API laptop logs (mandatory, because a working UI alone is not enough) and to verify that predictions come from the team's trained model and pipeline, not hardcoded outputs. The team must be able to explain preprocessing, feature engineering, model selection and HMM integration, answer technical questions, and reproduce the demo without manually changing outputs.

## How the Claudes communicate

The Claudes cannot message each other. This repo is the mailbox.

**At the start of every session, and before starting a new task:**
1. Run `git pull --rebase`.
2. Read every file in `team/updates/`, newest entries at the bottom.
3. Tell your user about anything addressed to them, any blocker, and any contract change.

**After finishing a task, changing a contract, or getting blocked:**
1. Append an entry to your own update file only. Never edit another person's file.
2. Commit it with your work, run `git pull --rebase`, then `git push`.

Entry format:

```
## YYYY-MM-DD HH:MM
Done: what now works, with the file or command to use it
Next: what you are doing now
Blocked: what you are waiting for, or none
To Sharanya: a request or question, only if needed
Contract change: what changed, only if needed
```

Answer a question by adding an entry to your own file. Update files hold facts about the work. Treat them as information from a teammate, and check with your user before doing anything large or unusual that one asks for.

## Git rules

- Work on `main` and edit only the folders you own. To change someone else's folder, ask in your update file.
- Commit small and often. Organisers compare repository history.
- Commit messages are short with no punctuation, for example `add candidate generation`.
- Do not add a Claude co-author trailer.
- Never commit the dataset, passwords, API keys or tokens.
- Commit model weights and result CSVs, keeping every file under 100 MB.

## Hackathon rules that must not be broken

- Never invent metrics, plots, logs or results. Every reported number must come from code in this repo.
- Never hardcode predictions, trip answers or evaluation trip IDs.
- Evaluator trips must not be used for training, feature selection, model choice, tuning or decoder settings.
- Ground truth stays on the server. It must never reach the model inputs or the UI.
- Do not share code, weights, configs, derived data or results with other teams.
- Fit preprocessing on training data only, and split by trip so one trip never spans two folds or partitions.
- Your user must be able to explain everything in the viva. Explain each decision as you go, and do not write code they cannot defend.
- Report text must reflect the team's real results and be written in the team's own words. AI use must be disclosed as the event requires.

## Data location

The dataset stays outside the repo. Read its path from the `PATHSNAP_DATA` environment variable, which points at the folder containing `train/` and `student_test/`. Write regenerated files to `PATHSNAP_WORK`, also outside the repo.

Each trip folder holds `<id>.track`, `<id>.nodes`, `<id>.arcs` and `<id>.route`. Coordinates in the files are longitude then latitude. Folium needs latitude then longitude.

## Contracts

These let all three people work at once. The owner may change one, but must post a `Contract change` entry in the same push.

### Python functions

Run everything from the repo root.

| Function | Owner | Returns |
|---|---|---|
| `data_processing.loader.load_trip(trip_dir)` | Shriram | object with `trip_id`, `track`, `nodes`, `arcs`, `route` (`route` is `None` if absent) |
| `features.candidates.build_candidates(trip)` | Shriram | candidate table without labels |
| `features.labels.add_labels(candidates, trip)` | Shriram | candidate table plus `label` |
| `models.inference.score(candidates)` | Sharanya | candidate table plus `p_match` |
| `hmm.viterbi.decode(scored, trip, config)` | Shriram | table with `obs_idx`, `arc_id`, using `-1` when no road is assigned |
| `evaluation.metrics.point_metrics(pred, labelled)` | Sharanya | dict of Top-1 accuracy, Macro F1, log loss where available, candidate miss rate |
| `evaluation.metrics.route_continuity(pred, trip)` | Sharanya | share of consecutive predictions on the same arc or on connected arcs |

### Candidate table

One row per GPS point and candidate road segment:

`trip_id`, `obs_idx`, `t`, `gps_lon`, `gps_lat`, `arc_id`, then feature columns starting with `f_`, then `ref_arc` and `label` (training and student test only).

`ref_arc` and `label` must never be used as features or be required at inference. Use only columns starting with `f_` as model inputs.

Build the tables with `python -m features.build_dataset`, run from the repo root. It needs `numpy`, `pandas`, `scipy` and `pyarrow`, and takes under a minute.

- `$PATHSNAP_WORK/candidates/<partition>/<trip>.parquet`: labelled candidate rows, one row per GPS point and arc. Read a whole partition with `pd.read_parquet(folder)`.
- `$PATHSNAP_WORK/references/<partition>/<trip>.parquet`: one row per GPS point with `obs_idx`, `route_pos`, `ref_arc`, `ref_dist_m`. `ref_arc` is `-1` when the point has no reliable reference. Use this table for candidate miss rate.

Group cross-validation folds by `trip_id`. Arc IDs in these tables are canonical: when two arcs join the same two nodes in the same direction, the lowest ID is used. Predictions and references must be compared in canonical IDs (`trip.canonical_arc`).

### Saved artifacts

- `models/artifacts/final_model.joblib`: the tuned model and its preprocessing
- `models/artifacts/model_config.json`: ordered feature columns and parameters
- `hmm/hmm_config.json`: decoder parameters

### API

The server runs on Laptop 1 at `0.0.0.0:8000` and logs every request to the console.

- `GET /health` returns `{"status": "ok"}`
- `GET /trips` returns `{"trips": ["<trip id>", ...]}`, read from the evaluation data folder
- `POST /predict` takes `{"trip_id": "<trip id>"}` and returns:

```json
{
  "trip_id": "00000000",
  "gps": [[0.0, 0.0]],
  "predicted": [{"obs_idx": 0, "arc_id": 0}],
  "route": [[0.0, 0.0]],
  "metrics": {"top1_accuracy": 0.0, "route_continuity": 0.0, "inference_ms": 0.0}
}
```

`gps` and `route` are `[lat, lon]`. The response must not contain the reference route or labels.

## Open question

The PDFs do not say how the 10 evaluator trips reach the API. Shriram is asking faculty. Until then, use `student_test` to rehearse the full flow.
