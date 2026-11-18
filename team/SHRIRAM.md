# Shriram's brief: data, features and route reconstruction

Read `CLAUDE.md` first. Everyone else depends on your candidate table, so ship a simple version early and improve it afterwards.

## Build, in this order

1. `data_processing/`: load and validate trips, handle bad points, timestamp gaps and unsorted times, and convert coordinates to metres.
2. `features/labels.py`: `.route` gives arcs in order but has no per-point labels. Design and justify how each GPS point gets its reference arc, and record the limits of that method.
3. `features/candidates.py`: generate candidate arcs per GPS point. Measure the candidate miss rate and how it changes with search radius or candidate count.
4. `features/`: engineer the `f_` features and a script that builds the training and student-test tables.
5. `hmm/`: hidden states, emissions from `p_match`, transitions from road connectivity, Viterbi, missing candidates, GPS gaps and disconnected segments.
6. Tune decoder settings on training and student-test data only, and compare classifier-only against classifier plus HMM, including cases where the HMM did not help.

## Tell the team when

- The first candidate table works: tell Sharanya how to generate it.
- `decode` works, even on rough scores: tell Shriya.

## Also yours

- Create the GitHub repo and add Sharanya and Shriya.
- Ask faculty how evaluator trips reach the API, then post the answer.
- Make the repo public at submission.

## Report sections

The preprocessing and feature parts of 2.2, and 5 HMM and Viterbi. Abdullah owns the EDA code, writes the EDA part of 2.2 and assembles the final PDF. Use his route-fit results when checking labels.

## Viva

Be ready to explain the label method, the distance calculation, each feature, the Viterbi recurrence and how gaps are handled.
