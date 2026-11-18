"""Save the final random forest's impurity importance for each feature and group.

This describes what the fitted model relied on. No feature was removed because of it.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

ARTIFACTS = Path("models") / "artifacts"
OUTPUT = Path(__file__).parent / "outputs" / "feature_importance.csv"
GROUPS = {
    "distance": ["f_dist_m", "f_dist_rank", "f_dist_gap_to_best_m", "f_dist_ratio_to_best"],
    "position on arc": ["f_signed_offset_m", "f_along_arc", "f_dist_to_arc_end_m", "f_arc_length_log"],
    "heading": ["f_heading_cos", "f_heading_known", "f_heading_rank", "f_heading_cos_carried", "f_heading_cos_next"],
    "movement": ["f_straightness", "f_net_speed_mps", "f_speed_mps", "f_step_prev_m", "f_dt_prev_log", "f_dt_next_log"],
    "neighbouring points": ["f_context_dist_mean_m", "f_context_dist_max_m", "f_anchor_match", "f_anchor_support"],
    "road connectivity": ["f_connected_prev", "f_connected_next", "f_from_out_degree", "f_to_out_degree", "f_has_reverse"],
    "candidate count": ["f_n_candidates"],
}


def main():
    model = joblib.load(ARTIFACTS / "final_model.joblib")
    columns = json.loads((ARTIFACTS / "model_config.json").read_text(encoding="utf-8"))["feature_columns"]
    group_of = {name: group for group, names in GROUPS.items() for name in names}
    table = pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
    table["group"] = table["feature"].map(group_of)
    assert table["group"].notna().all()
    table["group_importance"] = table.groupby("group")["importance"].transform("sum")
    table = table.sort_values("importance", ascending=False)
    table.to_csv(OUTPUT, index=False, float_format="%.5f")
    print(table.head(10).round(4).to_string(index=False))
    print(table.groupby("group")["importance"].sum().sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
