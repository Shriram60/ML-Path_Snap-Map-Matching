"""
############################################################################
#  PLACEHOLDER PIPELINE - DEVELOPMENT ONLY - DELETE THIS FILE BEFORE THE DEMO
#
#  Temporary stand-in for Sharanya's models.inference.score, so the API and
#  UI can be tested end to end before the trained model is exported.
#    score   p_match = Gaussian of f_dist_m. No model, nothing learned.
#  The API only uses a function from here if the real one cannot be
#  imported AND PATHSNAP_PLACEHOLDER=1 is set. Every response made with it
#  is flagged in the console and in the UI.
#  Never report any number produced with this file.
############################################################################
"""
import numpy as np

SIGMA_M = 10.0


def score(candidates):
    out = candidates.copy()
    out["p_match"] = np.exp(-0.5 * (out["f_dist_m"].to_numpy(dtype=float) / SIGMA_M) ** 2)
    return out
