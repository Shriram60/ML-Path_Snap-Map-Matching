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
