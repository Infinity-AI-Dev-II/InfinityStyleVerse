import pandas as pd
from typing import Dict, List, Optional

class StratifiedAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def analyze(self, stratify_by: str, agg: str = "weighted") -> Dict:
        results = {}
        strata = self.data[stratify_by].unique()

        per_stratum = []

        for group in strata:
            subset = self.data[self.data[stratify_by] == group]
            control = subset[subset["treatment"] == 0]["outcome"]
            treatment = subset[subset["treatment"] == 1]["outcome"]

            mean_control = control.mean()
            mean_treatment = treatment.mean()
            uplift = mean_treatment - mean_control

            per_stratum.append({
                "stratum": group,
                "n_control": len(control),
                "n_treatment": len(treatment),
                "mean_control": mean_control,
                "mean_treatment": mean_treatment,
                "uplift": uplift
            })

        per_stratum_df = pd.DataFrame(per_stratum)

        if agg == "weighted":
            # weight by sample size
            total = self.data.groupby("treatment")["outcome"].mean()
            overall_uplift = total[1] - total[0]
        else:
            # average across strata equally
            overall_uplift = per_stratum_df["uplift"].mean()

        results["per_stratum"] = per_stratum_df
        results["overall_uplift"] = overall_uplift

        return results
