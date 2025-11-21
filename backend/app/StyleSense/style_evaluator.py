#This code is an evaluation scaffold for a two-task model:
#for regression and classification
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
from typing import Dict, Union
import numpy as np
import json
#This function evaluates model predictions for two tasks:
# Aesthetic scoring (continuous values, e.g., 0–1)
# Occasion classification (categorical labels like "party", "formal")
def evaluate_models(
    y_true_aesthetic: np.ndarray,
    y_pred_aesthetic: np.ndarray,
    y_true_occasion: np.ndarray,
    y_pred_occasion: np.ndarray,
) -> Dict[str, Union[float, bool, str]]:
    results = {}#Stores all metrics and evaluation info.
    
    #kendalltau computes a correlation coefficient between true scores and predicted scores.
    tau, _ = kendalltau(y_true_aesthetic, y_pred_aesthetic)
    results['aesthetic_kendall_tau'] = float(f"{tau:.4f}")
    # computes the macro-averaged F1 score across all classes.
    #Macro averaging treats all classes equally, regardless of frequency.
    results['occasion_macro_f1'] = float(f"{f1_score(y_true_occasion, y_pred_occasion, average='macro'):.4f}")
    
    results['tau_gate_passed'] = results['aesthetic_kendall_tau'] >= 0.40
    results['occasion_f1_gate_passed'] = results['occasion_macro_f1'] >= 0.80

    results['status'] = "Evaluation Scaffolding is functional."

    return results

if __name__ == '__main__':
    dummy_true_scores = np.array([0.9, 0.5, 0.7, 0.8])
    dummy_pred_scores = np.array([0.8, 0.6, 0.8, 0.9])
    dummy_true_labels = np.array(['party', 'casual', 'party', 'formal'])
    dummy_pred_labels = np.array(['party', 'casual', 'formal', 'formal'])

    print("\nRunning initial evaluation scaffolding test")
    sim_results = evaluate_models(dummy_true_scores, dummy_pred_scores, dummy_true_labels, dummy_pred_labels)
    print(json.dumps(sim_results, indent=2))