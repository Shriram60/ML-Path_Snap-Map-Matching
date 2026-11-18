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
## 2026-11-18 10:45
Done: labels now use driving direction, and stopped points on out-and-back roads have no reference because their direction cannot be known. Added 4 features for stops and carried heading, so there are now 29 `f_` features
Next: rerun decoder tuning when the final model exists
Blocked: final `p_match` scores from Sharanya
To Sharanya: pull and rerun `python -m features.build_dataset` before training. Any table built before this entry has old labels and must be rebuilt. Read feature names from columns starting with `f_` rather than hardcoding them
To Shriya: no API change, but rebuild if you have a local table
To Abdullah: stopped vehicles show large apparent speed from GPS jitter, for example in trip 00000077. This is worth covering in your GPS quality analysis
Contract change: label definition and feature list changed, column layout is otherwise the same
## 2026-11-18 10:56
Done: faculty confirmed that hyperparameter tuning must be Bayesian only, with no grid or random search. Candidate settings and decoder settings are now tuned with Optuna. Candidates are now within 50 m, up to 24 per point
Next: rerun decoder tuning when the final model exists
Blocked: final `p_match` scores from Sharanya
To Sharanya: pull and rerun `python -m features.build_dataset` again before training, because the candidate radius changed. Use Bayesian optimisation only, and do not use `GridSearchCV` or `RandomizedSearchCV` anywhere
To Shriya: add `optuna` to the requirements. Arc `-1` now means no road within 50 m of that GPS point
To Abdullah: the report must describe Bayesian tuning only
Contract change: candidate radius is 50 m. Rule added to `CLAUDE.md`