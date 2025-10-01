import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, Tuple


class Calibration:
    def __init__(self):
        self.platt_model = None
        self.iso_model = None

    def fit_platt(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        y_pred = y_pred.reshape(-1, 1)
        self.platt_model = LogisticRegression(solver="lbfgs")
        self.platt_model.fit(y_pred, y_true)

    def predict_platt(self, y_pred: np.ndarray) -> np.ndarray:
        if self.platt_model is None:
            raise ValueError("Platt model not fitted yet. Call fit_platt() first.")
        return self.platt_model.predict_proba(y_pred.reshape(-1, 1))[:, 1]

    def fit_isotonic(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        self.iso_model = IsotonicRegression(out_of_bounds="clip")
        self.iso_model.fit(y_pred, y_true)

    def predict_isotonic(self, y_pred: np.ndarray) -> np.ndarray:
        if self.iso_model is None:
            raise ValueError("Isotonic model not fitted yet. Call fit_isotonic() first.")
        return self.iso_model.predict(y_pred)

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        return {
            "brier_score": brier_score_loss(y_true, y_pred),
            "log_loss": log_loss(y_true, y_pred)
        }


# Example usage (for demo / unit test)
if __name__ == "__main__":
    np.random.seed(42)

    # Simulate uncalibrated probabilities
    y_true = np.random.binomial(1, 0.3, size=1000)  # true click ~30%
    y_pred_uncal = np.random.uniform(0, 1, size=1000)  # uncalibrated model scores

    calib = Calibration()

    # Platt Scaling 
    calib.fit_platt(y_pred_uncal, y_true)
    y_pred_platt = calib.predict_platt(y_pred_uncal)
    print("Platt Scaling Eval:", Calibration.evaluate(y_true, y_pred_platt))

    # Isotonic Regression 
    calib.fit_isotonic(y_pred_uncal, y_true)
    y_pred_iso = calib.predict_isotonic(y_pred_uncal)
    print("Isotonic Regression Eval:", Calibration.evaluate(y_true, y_pred_iso))
