import numpy as np
import pandas as pd
from typing import Dict, Optional


class CounterfactualEvaluator:
    def __init__(self, clip: Optional[float] = None):
           self.clip = clip
    
    def _clip_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.clip is not None:
            return np.minimum(weights, self.clip)
        return weights

    def ips(self, rewards: np.ndarray, propensities: np.ndarray, new_policy_probs: np.ndarray) -> Dict:
        # Compute weights = new policy prob / old policy prob
        weights = new_policy_probs / propensities
        weights = self._clip_weights(weights)

        ips_estimates = rewards * weights
        estimate = np.mean(ips_estimates)
        variance = np.var(ips_estimates) / len(rewards)

        return {"ips_estimate": estimate, "ips_variance": variance}

    def dr(
        self,
        rewards: np.ndarray,
        propensities: np.ndarray,
        new_policy_probs: np.ndarray,
        reward_model_preds: np.ndarray
    ) -> Dict:
        weights = new_policy_probs / propensities
        weights = self._clip_weights(weights)

        # DR formula: model prediction + weight * (observed - predicted)
        dr_estimates = reward_model_preds + weights * (rewards - reward_model_preds)

        estimate = np.mean(dr_estimates)
        variance = np.var(dr_estimates) / len(rewards)

        return {"dr_estimate": estimate, "dr_variance": variance}
