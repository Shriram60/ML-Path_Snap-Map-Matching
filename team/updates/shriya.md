# Shriya's updates

Only Shriya's Claude edits this file. Append new entries at the bottom.

## 2026-11-18 10:45
Done: API, UI, Folium maps and two-laptop deployment run end to end on all 10 student_test trips, using a temporary placeholder pipeline.
- API: `uvicorn api.main:app --host 0.0.0.0 --port 8000` from the repo root. It imports every contract function from your modules. If one is missing it refuses to start, unless `PATHSNAP_PLACEHOLDER=1` is set. Then it uses `api/placeholder.py` (a nearest-road rule, no model, no HMM) for the missing ones only, and flags it in the console, in a response header and as a red banner in the UI. Your real functions are picked up as soon as they are pulled, with no edits to the API. The placeholder file will be deleted before the demo.
- Ground truth: the API passes a copy of the trip with `route=None` to build_candidates, score and decode. The real route is used only afterwards, server-side, for add_labels and point_metrics. It is never returned.
- UI: `deployment\run_ui.bat <laptop-1-ip>`, then open the URL it prints. The API address is configurable. Handles API down, timeouts (Retry) and error details.
- Maps: `python -m visualisations.route_map --all --partition student_test` writes to `visualisations/output/` (not committed). It uses the same pipeline as the API.
- Steps for both laptops, firewall and demo checklist: `deployment/README.md`.
Next: switching to the real pipeline as your modules land, then the two-laptop test and report screenshots.
Blocked: the real pipeline functions, `models/artifacts/final_model.joblib` and `hmm/hmm_config.json`. No placeholder number is reported anywhere.
To Shriram: the API relies on these points, so please keep them or tell me. (1) `trip.track`, `trip.nodes` and `trip.arcs` convert with `np.asarray` to columns lon, lat, t / lon, lat / from_node, to_node. (2) The trip object is a normal or non-frozen dataclass so the API can set `route=None` on a copy. (3) `decode` returns one row per GPS point and never reads `trip.route`. (4) `hmm/hmm_config.json` exists, because the API passes it to `decode`. Also: nobody owns a root README. Should you write it, or should I turn `deployment/README.md` into one?
To Sharanya: (1) Please have `score` load the joblib once at module level, not on every call. The API runs a warm-up score at startup. (2) The API reads `top1_accuracy`, `macro_f1` and `candidate_miss_rate` from `point_metrics`, so please keep those key names. (3) Pin your scikit-learn and joblib versions in `deployment/requirements-api.txt`. I will add them if you post them here.
Contract change: `/predict` `metrics` now also has `macro_f1` and `candidate_miss_rate`. Any metric is `null` when it is unavailable, for example `top1_accuracy` if a trip has no reference route. Responses carry the header `X-PathSnap-Pipeline: trained` or `placeholder`. Abdullah, `qa/test_api.py` should allow nulls and fail if that header says `placeholder` on demo day.

## 2026-11-18 10:35
Done: the API now runs Shriram's real `load_trip`, `build_candidates` and `add_labels` on all 10 student_test trips (0.05 to 0.9 s each), and no response contains `label`, `ref_arc` or reference data. Inference uses `trip.without_route()`, and `ref_arc` and `label` are dropped before `score` as an extra guard. `gps` and `predicted` are now aligned by position (`predicted[i]` belongs to `gps[i]`, one entry per cleaned GPS point, `-1` when unassigned), so this still works if the loader drops bad points. Maps and UI updated to match. `pyarrow` added to `deployment/requirements-api.txt`.
Next: switching `score`, `decode`, `point_metrics` and `route_continuity` to the real ones as they land. These four are the only placeholders left.
Blocked: `models.inference.score`, `hmm.viterbi.decode`, `evaluation.metrics`, `models/artifacts/final_model.joblib`, `hmm/hmm_config.json`.
To Shriram: thanks, the loader worked with no changes on your side. The API compares predictions to `ref_arc` from `add_labels`, assuming both are canonical arc IDs. Please confirm `decode` returns canonical IDs, as candidates do.
To Sharanya: `point_metrics(pred, labelled)` gets `labelled` straight from `add_labels`, which keeps only points with a reliable reference and includes `ref_arc`. For candidate miss rate, please compute it the way CLAUDE.md says (references table) inside your function. My placeholder version reports 0 for every trip because `add_labels` drops points that have no candidates.
Contract change: `/predict` `predicted` has exactly one entry per returned `gps` point, in the same order.

## 2026-11-18 10:45
Done: the API now uses Shriram's real `hmm.viterbi.decode` (config loaded once at startup with `load_config`), and computes metrics with `features.labels.reference_table` and `hmm.route_metrics.score_trip` after prediction, on the server only. All 10 student_test trips run in 0.1 to 1.9 s round trip, the UI and maps work, and no response contains reference data. `/predict` metrics are `top1_accuracy`, `route_continuity` (strict), `route_f1`, `candidate_miss_rate` and `inference_ms`.
Next: switching `score` to the trained model when it is exported. It is the only placeholder left.
Blocked: `models.inference.score` and `models/artifacts/final_model.joblib` from Sharanya.
To Sharanya: the API calls `score(candidates)` on the table from `build_candidates` with no `ref_arc` or `label`. Please return it with `p_match` added and all other columns kept, because `decode` needs `f_along_arc`. Please also load the model once at module level.
Contract change: the API no longer calls `add_labels`, `evaluation.metrics.point_metrics` or `route_continuity`. `macro_f1` is replaced by `route_f1` in the `/predict` metrics.

## 2026-11-18 11:00
Done: added optuna to deployment/requirements-api.txt for the tuning scripts.
Next: switching score to the trained model when Sharanya exports it.
Blocked: models.inference.score and models/artifacts/final_model.joblib.
