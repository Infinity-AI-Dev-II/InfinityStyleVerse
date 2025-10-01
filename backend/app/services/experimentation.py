import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
from .mde_calculator import MDECalculator, quick_sample_size
try:
    from .sequential_monitor import SequentialMonitor, ExperimentConfig, StoppingDecision
except ImportError:
    try:
        from sequential_monitor import SequentialMonitor, ExperimentConfig, StoppingDecision
    except ImportError:
        print("Warning: Sequential monitor not available")

def estimate_cuped_correlation(
    control_metric: np.ndarray,
    control_covariate: np.ndarray,
    treatment_metric: np.ndarray,  
    treatment_covariate: np.ndarray
) -> float:
    """Estimate correlation for MDE planning from pilot data."""
    X = np.concatenate([control_covariate, treatment_covariate])
    Y = np.concatenate([control_metric, treatment_metric])
    return np.corrcoef(X, Y)[0, 1]

def plan_experiment_from_pilot(
    pilot_results: dict,
    target_uplift: float,
    power: float = 0.8,
    alpha: float = 0.05
) -> dict:
    """Plan experiment using pilot CUPED results."""
    correlation = pilot_results.get('covariate_outcome_corr', 0)
    baseline_mean = pilot_results.get('control_mean_raw', 0)
    
    # Estimate if binary or continuous
    is_binary = 0 <= baseline_mean <= 1
    
    calc = MDECalculator()
    
    if is_binary:
        return calc.calculate_sample_size_binary(
            mde_relative=target_uplift,
            baseline_rate=baseline_mean,
            power=power,
            alpha=alpha,
            variance_reduction=correlation**2
        )
    else:
        # For continuous, need to estimate std
        pooled_std = np.sqrt(pilot_results.get('var_control_raw', 1))
        return calc.calculate_sample_size_continuous(
            mde=target_uplift * baseline_mean,  # Absolute effect
            pooled_std=pooled_std,
            power=power,
            alpha=alpha,
            variance_reduction=correlation**2
        )

def calculate_cuped_adjusted_metric(
    control_metric: np.ndarray,
    control_covariate: np.ndarray,
    treatment_metric: np.ndarray,
    treatment_covariate: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Apply CUPED adjustment with proper statistical inference.
    
    Args:
        control_metric: Array of observed outcomes for control group
        control_covariate: Pre-experiment covariates for control group  
        treatment_metric: Array of observed outcomes for treatment group
        treatment_covariate: Pre-experiment covariates for treatment group
        alpha: Significance level for confidence intervals
        
    Returns:
        Dictionary with adjusted means, statistics, and confidence intervals
    """
    
    # Input validation
    if len(control_metric) != len(control_covariate):
        raise ValueError("Control metric and covariate arrays must have same length")
    if len(treatment_metric) != len(treatment_covariate):
        raise ValueError("Treatment metric and covariate arrays must have same length")
    
    # Combine data for theta estimation
    X = np.concatenate([control_covariate, treatment_covariate])
    Y = np.concatenate([control_metric, treatment_metric])
    
    # Estimate theta (using population estimates - ddof=0)
    cov_xy = np.cov(X, Y, ddof=0)[0, 1]
    var_x = np.var(X, ddof=0)
    
    if var_x == 0:
        # If no variance in covariate, CUPED provides no benefit
        theta = 0
        print("Warning: Zero variance in covariate. CUPED adjustment has no effect.")
    else:
        theta = cov_xy / var_x
    
    # Calculate pooled covariate mean
    x_mean = np.mean(X)
    
    # Apply CUPED adjustment
    control_adj = control_metric - theta * (control_covariate - x_mean)
    treatment_adj = treatment_metric - theta * (treatment_covariate - x_mean)
    
    # Calculate means
    control_mean_raw = np.mean(control_metric)
    treatment_mean_raw = np.mean(treatment_metric)
    control_mean_cuped = np.mean(control_adj)
    treatment_mean_cuped = np.mean(treatment_adj)
    
    # Calculate uplifts
    uplift_raw = treatment_mean_raw - control_mean_raw
    uplift_cuped = treatment_mean_cuped - control_mean_cuped
    
    # Calculate variances and standard errors
    var_control_raw = np.var(control_metric, ddof=1)
    var_treatment_raw = np.var(treatment_metric, ddof=1)
    var_control_cuped = np.var(control_adj, ddof=1)
    var_treatment_cuped = np.var(treatment_adj, ddof=1)
    
    # Standard errors for uplift
    se_raw = np.sqrt(var_control_raw / len(control_metric) + 
                     var_treatment_raw / len(treatment_metric))
    se_cuped = np.sqrt(var_control_cuped / len(control_adj) + 
                       var_treatment_cuped / len(treatment_adj))
    
    # T-test statistics
    if se_raw > 0:
        t_stat_raw = uplift_raw / se_raw
        df_raw = len(control_metric) + len(treatment_metric) - 2
        p_value_raw = 2 * (1 - stats.t.cdf(abs(t_stat_raw), df_raw))
        
        # Confidence intervals for raw uplift
        t_critical_raw = stats.t.ppf(1 - alpha/2, df_raw)
        ci_raw_lower = uplift_raw - t_critical_raw * se_raw
        ci_raw_upper = uplift_raw + t_critical_raw * se_raw
    else:
        t_stat_raw = p_value_raw = float('nan')
        ci_raw_lower = ci_raw_upper = uplift_raw
    
    if se_cuped > 0:
        t_stat_cuped = uplift_cuped / se_cuped
        df_cuped = len(control_adj) + len(treatment_adj) - 2
        p_value_cuped = 2 * (1 - stats.t.cdf(abs(t_stat_cuped), df_cuped))
        
        # Confidence intervals for CUPED uplift
        t_critical_cuped = stats.t.ppf(1 - alpha/2, df_cuped)
        ci_cuped_lower = uplift_cuped - t_critical_cuped * se_cuped
        ci_cuped_upper = uplift_cuped + t_critical_cuped * se_cuped
    else:
        t_stat_cuped = p_value_cuped = float('nan')
        ci_cuped_lower = ci_cuped_upper = uplift_cuped
    
    # Variance reduction calculation
    var_pooled_raw = np.var(Y, ddof=1)
    var_pooled_cuped = np.var(np.concatenate([control_adj, treatment_adj]), ddof=1)
    
    if var_pooled_raw > 0:
        var_reduction = (var_pooled_raw - var_pooled_cuped) / var_pooled_raw
    else:
        var_reduction = 0
    
    # Relative efficiency (how much smaller sample size needed with CUPED)
    if se_raw > 0:
        relative_efficiency = (se_raw / se_cuped) ** 2 if se_cuped > 0 else float('inf')
    else:
        relative_efficiency = 1.0
    
    return {
        # Theta and adjustment info
        "theta": theta,
        "covariate_mean": x_mean,
        "covariate_outcome_corr": np.corrcoef(X, Y)[0, 1],
        
        # Raw results
        "control_mean_raw": control_mean_raw,
        "treatment_mean_raw": treatment_mean_raw,
        "uplift_raw": uplift_raw,
        "se_raw": se_raw,
        "t_stat_raw": t_stat_raw,
        "p_value_raw": p_value_raw,
        "ci_raw": (ci_raw_lower, ci_raw_upper),
        
        # CUPED results  
        "control_mean_cuped": control_mean_cuped,
        "treatment_mean_cuped": treatment_mean_cuped,
        "uplift_cuped": uplift_cuped,
        "se_cuped": se_cuped,
        "t_stat_cuped": t_stat_cuped,
        "p_value_cuped": p_value_cuped,
        "ci_cuped": (ci_cuped_lower, ci_cuped_upper),
        
        # Efficiency metrics
        "var_reduction": var_reduction,
        "relative_efficiency": relative_efficiency,
        "se_reduction": (se_raw - se_cuped) / se_raw if se_raw > 0 else 0,
        
        # Sample sizes
        "n_control": len(control_metric),
        "n_treatment": len(treatment_metric),
        "n_total": len(control_metric) + len(treatment_metric)
    }


def validate_cuped_aa_test(
    n_simulations: int = 100,
    sample_size: int = 1000,
    alpha: float = 0.05,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Validate CUPED implementation with A/A test simulation.
    Should show no false positives (Type I error rate ≈ alpha).
    
    Args:
        n_simulations: Number of A/A tests to simulate
        sample_size: Sample size per group  
        alpha: Significance level
        random_seed: Random seed for reproducibility
        
    Returns:
        Validation results with Type I error rates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    p_values_raw = []
    p_values_cuped = []
    var_reductions = []
    
    for i in range(n_simulations):
        # Generate A/A test data (no treatment effect)
        pre_metric = np.random.normal(0, 1, size=2*sample_size)
        
        # Random assignment
        treatment = np.random.binomial(1, 0.5, size=2*sample_size)
        
        # Outcome correlated with pre-metric but NO treatment effect
        outcome = 0.5 + 0.3 * pre_metric + np.random.normal(0, 0.4, size=2*sample_size)
        
        # Split by treatment
        control_mask = treatment == 0
        treatment_mask = treatment == 1
        
        try:
            results = calculate_cuped_adjusted_metric(
                control_metric=outcome[control_mask],
                control_covariate=pre_metric[control_mask],
                treatment_metric=outcome[treatment_mask], 
                treatment_covariate=pre_metric[treatment_mask],
                alpha=alpha
            )
            
            p_values_raw.append(results["p_value_raw"])
            p_values_cuped.append(results["p_value_cuped"])
            var_reductions.append(results["var_reduction"])
            
        except Exception as e:
            print(f"Error in simulation {i}: {e}")
            continue
    
    # Calculate Type I error rates
    type_i_error_raw = np.mean(np.array(p_values_raw) < alpha)
    type_i_error_cuped = np.mean(np.array(p_values_cuped) < alpha)
    
    return {
        "n_simulations": len(p_values_raw),
        "alpha": alpha,
        "type_i_error_raw": type_i_error_raw,
        "type_i_error_cuped": type_i_error_cuped,
        "expected_type_i_error": alpha,
        "raw_within_bounds": abs(type_i_error_raw - alpha) < 2 * np.sqrt(alpha * (1-alpha) / len(p_values_raw)),
        "cuped_within_bounds": abs(type_i_error_cuped - alpha) < 2 * np.sqrt(alpha * (1-alpha) / len(p_values_raw)),
        "mean_var_reduction": np.mean(var_reductions),
        "p_values_raw": p_values_raw[:10],  # First 10 for inspection
        "p_values_cuped": p_values_cuped[:10]
    }


# Example usage and testing
if __name__ == "__main__":
    # Run A/A validation test
    print("Running A/A validation test...")
    validation_results = validate_cuped_aa_test(n_simulations=100, random_seed=42)
    
    print(f"Type I Error Rate (Raw): {validation_results['type_i_error_raw']:.3f}")
    print(f"Type I Error Rate (CUPED): {validation_results['type_i_error_cuped']:.3f}")
    print(f"Expected: {validation_results['alpha']:.3f}")
    print(f"Mean Variance Reduction: {validation_results['mean_var_reduction']:.3f}")
    print(f"Raw within bounds: {validation_results['raw_within_bounds']}")
    print(f"CUPED within bounds: {validation_results['cuped_within_bounds']}")

def create_sequential_cuped_experiment(
    experiment_id: str,
    pilot_cuped_results: Dict,
    target_effect_size: float,
    max_sample_size: int,
    power: float = 0.8,
    alpha: float = 0.05
) -> SequentialMonitor:
    """
    Create a sequential experiment using CUPED results from pilot data.
    
    Args:
        experiment_id: Unique identifier for the experiment
        pilot_cuped_results: Results from calculate_cuped_adjusted_metric on pilot data
        target_effect_size: Minimum effect size you want to detect
        max_sample_size: Maximum sample size per group
        power: Statistical power (default 0.8)
        alpha: Significance level (default 0.05)
    
    Returns:
        Configured SequentialMonitor instance
    """
    
    # Estimate variance reduction from pilot CUPED results
    variance_reduction = pilot_cuped_results.get('var_reduction', 0.0)
    
    # Adjust max sample size based on CUPED efficiency
    if variance_reduction > 0:
        efficiency_gain = pilot_cuped_results.get('relative_efficiency', 1.0)
        adjusted_max_sample = int(max_sample_size / efficiency_gain)
        print(f"CUPED efficiency gain: {efficiency_gain:.1f}x")
        print(f"Adjusted max sample size: {adjusted_max_sample:,} (was {max_sample_size:,})")
        max_sample_size = adjusted_max_sample
    
    config = ExperimentConfig(
        experiment_id=experiment_id,
        alpha=alpha,
        beta=1-power,
        max_sample_size=max_sample_size,
        min_effect_size=abs(target_effect_size),
        spending_function="obrien_fleming",  # Conservative early, liberal late
        futility_enabled=True
    )
    
    return SequentialMonitor(config)

def run_sequential_cuped_analysis(
    monitor: SequentialMonitor,
    control_outcome: np.ndarray,
    treatment_outcome: np.ndarray,
    control_covariate: np.ndarray,
    treatment_covariate: np.ndarray,
    analysis_time: Optional[datetime] = None
) -> Dict:
    """
    Run a sequential analysis with CUPED adjustment.
    
    Args:
        monitor: SequentialMonitor instance
        control_outcome: Control group outcomes
        treatment_outcome: Treatment group outcomes
        control_covariate: Control group covariates
        treatment_covariate: Treatment group covariates
        analysis_time: Time of analysis
    
    Returns:
        Complete analysis results with recommendations
    """
    
    # Conduct the interim analysis with CUPED
    analysis_result = monitor.conduct_interim_analysis(
        control_data=control_outcome,
        treatment_data=treatment_outcome,
        control_covariate=control_covariate,
        treatment_covariate=treatment_covariate,
        analysis_time=analysis_time
    )
    
    # Get CUPED-specific metrics for additional context
    cuped_results = calculate_cuped_adjusted_metric(
        control_metric=control_outcome,
        control_covariate=control_covariate,
        treatment_metric=treatment_outcome,
        treatment_covariate=treatment_covariate
    )
    
    # Combine results
    return {
        "sequential_analysis": analysis_result,
        "cuped_metrics": {
            "variance_reduction": cuped_results["var_reduction"],
            "relative_efficiency": cuped_results["relative_efficiency"],
            "theta": cuped_results["theta"],
            "correlation": cuped_results["covariate_outcome_corr"]
        },
        "experiment_status": monitor.get_experiment_status(),
        "recommendations": {
            "action": analysis_result.stopping_decision.value,
            "reasoning": analysis_result.recommendation,
            "next_steps": _generate_next_steps(analysis_result, monitor)
        }
    }

def _generate_next_steps(analysis_result, monitor: SequentialMonitor) -> str:
    """Generate actionable next steps based on analysis results"""
    
    decision = analysis_result.stopping_decision
    
    if decision == StoppingDecision.STOP_FOR_EFFICACY:
        if analysis_result.effect_estimate > 0:
            return f"""
             IMPLEMENT WINNING VARIANT:
            • Roll out treatment to 100% of traffic
            • Document effect size: {analysis_result.effect_estimate:.3f}
            • Monitor for 1-2 weeks to confirm sustained impact
            • Update baseline metrics for future experiments
            """
        else:
            return f"""
             STOP HARMFUL TREATMENT:
            • Immediately stop treatment rollout
            • Revert to control variant for all users
            • Investigate root cause of negative impact
            • Document learnings and adjust future hypotheses
            """
    
    elif decision == StoppingDecision.STOP_FOR_FUTILITY:
        return f"""
         REDESIGN EXPERIMENT:
        • Current approach unlikely to detect meaningful effect
        • Consider larger effect size or different intervention
        • May need {int(analysis_result.sample_size_current * 2):,}+ more users for current effect
        • Review hypothesis and experimental design
        """
    
    elif decision == StoppingDecision.STOP_FOR_SAFETY:
        return f"""
         EMERGENCY STOP:
        • Immediately halt all treatment exposure
        • Investigate safety concerns and user impact
        • Notify stakeholders and document incident
        • Review safety thresholds for future experiments
        """
    
    else:  # CONTINUE
        next_analysis = monitor._estimate_next_analysis_time()
        next_sample_needed = int(monitor.config.max_sample_size * monitor.information_fractions[len(monitor.analyses_completed)])
        current_sample = analysis_result.sample_size_current
        
        return f"""
         CONTINUE EXPERIMENT:
        • Current power: {analysis_result.power_current:.1%}
        • Need {next_sample_needed - current_sample:,} more users for next analysis
        • Next analysis scheduled: {next_analysis.strftime('%Y-%m-%d') if next_analysis else 'TBD'}
        • Monitor daily metrics for any concerning trends
        """

def validate_sequential_monitoring_aa(
    n_simulations: int = 100,
    sample_size_per_analysis: int = 1000,
    max_analyses: int = 5,
    alpha: float = 0.05
) -> Dict:
    """
    Validate sequential monitoring with A/A test simulation.
    Should maintain Type I error rate despite multiple looks.
    
    Args:
        n_simulations: Number of A/A simulations
        sample_size_per_analysis: Sample size added at each analysis
        max_analyses: Maximum number of interim analyses
        alpha: Target significance level
    
    Returns:
        Validation results showing Type I error control
    """
    
    false_positive_count = 0
    analysis_counts = []
    stopping_reasons = []
    
    for sim in range(n_simulations):
        np.random.seed(sim + 1000)  # Different seed for each simulation
        
        # Create sequential monitor for this simulation
        config = ExperimentConfig(
            experiment_id=f"aa_sim_{sim}",
            alpha=alpha,
            max_analyses=max_analyses,
            max_sample_size=sample_size_per_analysis * max_analyses,
            spending_function="obrien_fleming"
        )
        
        monitor = SequentialMonitor(config)
        
        # Run A/A test (no true effect)
        stopped_early = False
        final_analysis = None
        
        for analysis_num in range(1, max_analyses + 1):
            # Generate A/A data (same distribution for both groups)
            current_n = analysis_num * sample_size_per_analysis
            
            control_data = np.random.normal(0.5, 0.3, current_n)
            treatment_data = np.random.normal(0.5, 0.3, current_n)  # Same as control
            
            # Add correlated covariate for CUPED testing
            covariate_control = np.random.normal(0, 1, current_n)
            covariate_treatment = np.random.normal(0, 1, current_n)
            
            # Make outcomes slightly correlated with covariate
            control_data += 0.2 * covariate_control
            treatment_data += 0.2 * covariate_treatment
            
            try:
                result = monitor.conduct_interim_analysis(
                    control_data, treatment_data,
                    covariate_control, covariate_treatment
                )
                
                final_analysis = result
                
                if result.stopping_decision != StoppingDecision.CONTINUE:
                    stopped_early = True
                    break
                    
            except Exception as e:
                print(f"Error in simulation {sim}, analysis {analysis_num}: {e}")
                break
        
        # Record results
        if final_analysis:
            analysis_counts.append(final_analysis.analysis_number)
            stopping_reasons.append(final_analysis.stopping_decision.value)
            
            # Count false positives (stopping for efficacy in A/A test)
            if final_analysis.stopping_decision == StoppingDecision.STOP_FOR_EFFICACY:
                false_positive_count += 1
    
    # Calculate Type I error rate
    type_i_error_rate = false_positive_count / n_simulations
    
    # Analysis distribution
    analysis_distribution = {}
    for count in analysis_counts:
        analysis_distribution[count] = analysis_distribution.get(count, 0) + 1
    
    # Stopping reason distribution
    reason_distribution = {}
    for reason in stopping_reasons:
        reason_distribution[reason] = reason_distribution.get(reason, 0) + 1
    
    return {
        "n_simulations": n_simulations,
        "target_alpha": alpha,
        "observed_type_i_error": type_i_error_rate,
        "false_positive_count": false_positive_count,
        "type_i_controlled": type_i_error_rate <= alpha * 1.1,  # Allow 10% tolerance
        "mean_analyses": np.mean(analysis_counts),
        "analysis_distribution": analysis_distribution,
        "stopping_reason_distribution": reason_distribution,
        "early_stopping_rate": sum(1 for c in analysis_counts if c < max_analyses) / len(analysis_counts)
    }

def compare_sequential_vs_fixed(
    true_effects: List[float],
    sample_size_per_group: int = 5000,
    alpha: float = 0.05,
    power: float = 0.8,
    n_simulations: int = 50
) -> pd.DataFrame:
    """
    Compare sequential testing vs fixed sample size testing.
    
    Args:
        true_effects: List of true effect sizes to test
        sample_size_per_group: Fixed sample size for comparison
        alpha: Significance level
        power: Target power
        n_simulations: Number of simulations per effect size
    
    Returns:
        DataFrame comparing sequential vs fixed approaches
    """
    
    results = []
    
    for true_effect in true_effects:
        print(f"Testing effect size: {true_effect:.3f}")
        
        # Sequential testing results
        seq_sample_sizes = []
        seq_detected = 0
        seq_analysis_counts = []
        
        # Fixed testing results  
        fixed_detected = 0
        
        for sim in range(n_simulations):
            np.random.seed(sim + int(true_effect * 1000))
            
            # Sequential test
            try:
                sim_result = run_sequential_simulation(
                    true_effect=true_effect,
                    sample_size_per_analysis=sample_size_per_group // 5,  # 5 analyses max
                    max_analyses=5,
                    baseline_std=0.3
                )
                
                if sim_result['final_analysis']:
                    final = sim_result['final_analysis']
                    seq_sample_sizes.append(final.sample_size_current)
                    seq_analysis_counts.append(final.analysis_number)
                    
                    if final.stopping_decision == StoppingDecision.STOP_FOR_EFFICACY:
                        seq_detected += 1
            
            except Exception as e:
                print(f"Sequential simulation error: {e}")
                continue
            
            # Fixed sample test
            control_fixed = np.random.normal(0, 0.3, sample_size_per_group)
            treatment_fixed = np.random.normal(true_effect, 0.3, sample_size_per_group)
            
            # Simple t-test
            t_stat, p_val = stats.ttest_ind(treatment_fixed, control_fixed)
            if p_val < alpha:
                fixed_detected += 1
        
        # Calculate metrics
        seq_power = seq_detected / n_simulations
        fixed_power = fixed_detected / n_simulations
        
        avg_seq_sample = np.mean(seq_sample_sizes) if seq_sample_sizes else sample_size_per_group * 2
        avg_seq_analyses = np.mean(seq_analysis_counts) if seq_analysis_counts else 5
        
        sample_efficiency = (sample_size_per_group * 2) / avg_seq_sample if avg_seq_sample > 0 else 1
        
        results.append({
            "true_effect": true_effect,
            "sequential_power": seq_power,
            "fixed_power": fixed_power,
            "power_difference": seq_power - fixed_power,
            "avg_sequential_sample": avg_seq_sample,
            "fixed_sample": sample_size_per_group * 2,
            "sample_efficiency": sample_efficiency,
            "avg_analyses": avg_seq_analyses,
            "time_savings": 1 - (avg_seq_analyses / 5)  # Assuming 5 was max
        })
    
    return pd.DataFrame(results)

# Integration with existing MDE calculator
def plan_sequential_experiment_with_mde(
    baseline_rate: float,
    target_uplift: float,
    covariate_correlation: float = 0.0,
    daily_traffic: int = 1000,
    max_duration_days: int = 30,
    alpha: float = 0.05,
    power: float = 0.8
) -> Dict:
    """
    Plan a sequential experiment using MDE calculator integration.
    
    Args:
        baseline_rate: Baseline conversion rate
        target_uplift: Target relative uplift
        covariate_correlation: Expected correlation with covariate
        daily_traffic: Daily traffic available
        max_duration_days: Maximum experiment duration
        alpha: Significance level
        power: Target statistical power
    
    Returns:
        Complete experiment plan with sequential parameters
    """
    
    try:
        from .mde_calculator import MDECalculator
        calc = MDECalculator()
    except ImportError:
        return {"error": "MDE Calculator not available"}
    
    # Calculate required sample size with CUPED
    variance_reduction = covariate_correlation ** 2
    
    sample_result = calc.calculate_sample_size_binary(
        mde_relative=target_uplift,
        baseline_rate=baseline_rate,
        power=power,
        alpha=alpha,
        variance_reduction=variance_reduction
    )
    
    required_per_group = sample_result["sample_size_control"]
    max_feasible_per_group = (daily_traffic * max_duration_days) // 2
    
    # Check feasibility
    is_feasible = required_per_group <= max_feasible_per_group
    
    if is_feasible:
        # Plan sequential analyses
        max_analyses = min(5, max_duration_days // 7)  # Weekly analyses, max 5
        
        config = ExperimentConfig(
            experiment_id="planned_experiment",
            alpha=alpha,
            beta=1-power,
            max_sample_size=required_per_group,
            min_effect_size=baseline_rate * target_uplift,
            max_analyses=max_analyses,
            spending_function="obrien_fleming"
        )
        
        # Estimate duration with early stopping
        expected_duration = max_duration_days * 0.7  # Assume 30% early stopping savings
        
        return {
            "feasible": True,
            "required_sample_per_group": required_per_group,
            "max_feasible_per_group": max_feasible_per_group,
            "expected_duration_days": expected_duration,
            "max_analyses": max_analyses,
            "variance_reduction": variance_reduction,
            "cuped_sample_savings": sample_result.get("sample_size_without_cuped", required_per_group) - required_per_group,
            "sequential_config": config,
            "recommendations": f"""
             EXPERIMENT IS FEASIBLE:
            • Required sample: {required_per_group:,} per group
            • Expected duration: {expected_duration:.0f} days
            • Plan {max_analyses} interim analyses
            • CUPED expected to reduce sample by {variance_reduction:.1%}
            • Sequential testing may reduce duration by ~30%
            """
        }
    
    else:
        # Calculate achievable MDE with available traffic
        achievable_mde = calc.calculate_mde_binary(
            sample_size_per_group=max_feasible_per_group,
            baseline_rate=baseline_rate,
            power=power,
            alpha=alpha,
            variance_reduction=variance_reduction
        )
        
        return {
            "feasible": False,
            "required_sample_per_group": required_per_group,
            "max_feasible_per_group": max_feasible_per_group,
            "shortage": required_per_group - max_feasible_per_group,
            "achievable_mde": achievable_mde["mde_relative"],
            "target_mde": target_uplift,
            "recommendations": f"""
             EXPERIMENT NOT FEASIBLE WITH CURRENT TRAFFIC:
            • Need {required_per_group:,} per group, have {max_feasible_per_group:,}
            • Shortage: {required_per_group - max_feasible_per_group:,} users
            • Achievable MDE: {achievable_mde['mde_relative']:.1%} (target: {target_uplift:.1%})
            • Options: Increase duration, reduce target effect, or increase traffic
            """
        }

# Add to the main test section at the bottom of experimentation.py
if __name__ == "__main__":
    # ... existing CUPED tests ...
    
    # Test sequential monitoring integration
    print(f"\n Testing Sequential Monitoring Integration...")
    
    try:
        # Test A/A validation
        aa_validation = validate_sequential_monitoring_aa(
            n_simulations=20,  # Reduced for demo
            sample_size_per_analysis=500
        )
        
        print(f" Sequential A/A Validation:")
        print(f"   Target alpha: {aa_validation['target_alpha']:.3f}")
        print(f"   Observed Type I error: {aa_validation['observed_type_i_error']:.3f}")
        print(f"   Type I controlled: {aa_validation['type_i_controlled']}")
        print(f"   Mean analyses: {aa_validation['mean_analyses']:.1f}")
        
    except Exception as e:
        print(f"  Sequential monitoring test failed: {e}")
    
    print(f" All integration tests completed!")