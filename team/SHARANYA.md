# Sharanya's brief: models, tuning and evaluation

Read `CLAUDE.md` first. You turn Shriram's candidate table into the scores the HMM and API use.

## Start now, before real data is ready

Write `evaluation/metrics.py` and the training harness against the candidate table contract. Test them with a tiny made-up table, and never report anything from it. Switch to the real table as soon as Shriram posts it.

## Build, in this order

1. `evaluation/`: Top-1 accuracy per GPS point, Macro F1, precision, recall, log loss, candidate miss rate, route continuity, training time and inference time.
2. `models/`: train at least 5 distinct models from different algorithm families. Use grouped cross-validation by trip on the 80 training trips, and compare on the 10 student-test trips.
3. Choose the primary selection metric and justify it. Candidates are likely imbalanced, so check that and handle it deliberately. Probability quality matters because the HMM consumes the scores.
4. `tuning/`: Bayesian optimisation of the selected model with grouped cross-validation. Save the search space, objective, every trial and the compute cost.
5. Refit the tuned model, compare tuned against untuned on the same data, and export the artifacts.
6. `models/inference.py`: `score` must load the artifacts and work on tables with no `label` column.
7. `results/`: CSV logs, seeds, configurations and every comparison plot. Plot epoch-wise loss only for models that really produce it.

## Tell the team when

- A first model can score candidates: tell Shriram and Shriya, even if it is not final.
- Final artifacts are pushed: tell both again.
- A metric needs a different table or column: tell Shriram.

## Report sections

3 Model experimentation and comparison, including the required answers for every model, and 4 Bayesian optimisation.

## Viva

Be ready to explain why each model was tried, why the final one won, whether any overfit, the tuning objective, and what the metrics mean.
