# Shriram's updates

Only Shriram's Claude edits this file. Append new entries at the bottom.

## 2026-11-18 10:24
Done: `load_trip`, `build_candidates` and `add_labels` work. Run `python -m features.build_dataset` to create the labelled tables in `$PATHSNAP_WORK` (about 1.83M train rows and 223k student-test rows, about 10 percent positive)
Next: HMM and Viterbi in `hmm/`
Blocked: none
To Sharanya: the candidate table is ready, use only `f_` columns as inputs and group folds by `trip_id`. Trip 00000033 is left out of training because its labels failed the heading check, so training has 79 trips. See `features/outputs/build_summary.csv`
To Shriya: `load_trip` works without a `.route` file, so evaluation trips can go through `build_candidates(trip)` directly. API dependencies from my side are `numpy`, `pandas`, `scipy`, `pyarrow`
To Abdullah: GPS continuing past the route is confirmed in trips 00000002, 00000033 and 00000054. Please send any others you find from `route_fit.csv`
Contract change: candidate table now includes `ref_arc`, arc IDs are canonical, and the build output locations are listed in `CLAUDE.md`

## 2026-11-18 10:31
Done: `hmm.viterbi.decode(scored, trip)` and `classifier_only(scored, trip)` work, about 2 seconds or less per student-test trip. `scored` needs `obs_idx`, `arc_id`, `f_along_arc` and `p_match`. `hmm/hmm_config.json` loads by default
Next: rerun decoder tuning when the final model exists, then write report sections
Blocked: final `p_match` scores from Sharanya
To Sharanya: when the model is exported, tell me. All current HMM numbers and `hmm/hmm_config.json` use a temporary baseline model, so they must not go in the report or model comparison
To Shriya: the API flow is `load_trip`, `build_candidates`, `models.inference.score`, `decode`. Arc `-1` means no road within 60 m of that GPS point. Keep references server-side, and use `features.labels.reference_table` with `hmm.route_metrics.score_trip` for metrics
To Abdullah: `hmm/outputs/hmm_comparison.csv` is temporary, wait for the final version before making report figures