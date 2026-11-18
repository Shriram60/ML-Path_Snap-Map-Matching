# Sharanya's updates

Only Sharanya's Claude edits this file. Append new entries at the bottom.



\## 2026-11-18 12:15

Done: 5 models compared with 5-fold grouped CV by trip (python -m models.train\_compare), results in results/model\_comparison.csv and results/plots. Random Forest selected on CV Top-1 0.9416 and CV log loss 0.0321. Bayesian tuning with Optuna TPE, 12 trials (python -m tuning.bayes\_opt --model random\_forest --timeout 900 --search-frac 0.05), results in results/tuning. Tuned model: student-test Top-1 0.9418, log loss 0.0271. Final model exported to models/artifacts

Next: report sections 3 and 4

Blocked: none

To Shriram: final model is ready. models.inference.score(candidates) adds p\_match, so you can re-tune the decoder on it now

To Shriya: replace the placeholder with models.inference.score(candidates). It loads models/artifacts/final\_model.joblib and model\_config.json, and works without label or ref\_arc

To Abdullah: report charts can read results/model\_comparison.csv and results/tuning/random\_forest\_tuned\_vs\_untuned.csv

