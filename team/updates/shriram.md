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
