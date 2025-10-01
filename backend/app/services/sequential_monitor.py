import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from enum import Enum

class StoppingDecision(Enum):
    CONTINUE = "continue"
    STOP_FOR_EFFICACY = "stop_for_efficacy"
    STOP_FOR_FUTILITY = "stop_for_futility"
    STOP_FOR_SAFETY = "stop_for_safety"

@dataclass
class AnalysisResult:
    """Result of an interim analysis"""
    analysis_number: int
    information_fraction: float
    test_statistic: float
    p_value: float
    effect_estimate: float
    confidence_interval: Tuple[float, float]
    stopping_decision: StoppingDecision
    efficacy_boundary: float
    futility_boundary: Optional[float]
    sample_size_current: int
    power_current: float
    recommendation: str

@dataclass  
class ExperimentConfig:
    """Configuration for sequential experiment"""
    experiment_id: str
    alpha: float = 0.05
    beta: float = 0.2  # 1 - power
    max_analyses: int = 5
    max_sample_size: int = 10000
    min_effect_size: float = 0.01
    spending_function: Literal["obrien_fleming", "pocock", "haybittle_peto"] = "obrien_fleming"
    futility_enabled: bool = True
    safety_threshold: float = None  # Stop if effect is significantly negative

class SequentialMonitor:
    """
    Sequential monitoring system with multiple spending functions and CUPED integration.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.analyses_completed = []
        self.experiment_stopped = False
        self.stopping_reason = None
        
        # Calculate spending boundaries upfront
        self.information_fractions = self._get_planned_information_fractions()
        self.efficacy_boundaries = self._calculate_efficacy_boundaries()
        self.futility_boundaries = self._calculate_futility_boundaries() if config.futility_enabled else None
        
    def _get_planned_information_fractions(self) -> np.ndarray:
        """Get planned information fractions for interim analyses"""
        return np.linspace(1/self.config.max_analyses, 1.0, self.config.max_analyses)
    
    def _calculate_efficacy_boundaries(self) -> np.ndarray:
        """Calculate efficacy boundaries using specified spending function"""
        alpha = self.config.alpha
        K = self.config.max_analyses
        t = self.information_fractions
        
        if self.config.spending_function == "obrien_fleming":
            return self._obrien_fleming_boundaries(alpha, t)
        elif self.config.spending_function == "pocock":
            return self._pocock_boundaries(alpha, K)
        elif self.config.spending_function == "haybittle_peto":
            return self._haybittle_peto_boundaries(alpha, K)
        else:
            raise ValueError(f"Unknown spending function: {self.config.spending_function}")
    
    def _obrien_fleming_boundaries(self, alpha: float, t: np.ndarray) -> np.ndarray:
        """O'Brien-Fleming boundaries (conservative early, liberal late)"""
        # Alpha spending function: alpha * (2 - 2*Φ(Φ^(-1)(1-α/2)/√t))
        z_alpha_2 = stats.norm.ppf(1 - alpha/2)
        
        boundaries = []
        cumulative_alpha = 0
        
        for i, t_i in enumerate(t):
            # Calculate cumulative alpha spent up to analysis i
            alpha_spent = 2 * (1 - stats.norm.cdf(z_alpha_2 / np.sqrt(t_i)))
            
            # Incremental alpha for this analysis
            if i == 0:
                incremental_alpha = alpha_spent
            else:
                incremental_alpha = alpha_spent - cumulative_alpha
            
            # Boundary for this analysis
            if incremental_alpha > 0:
                boundary = stats.norm.ppf(1 - incremental_alpha/2)
            else:
                boundary = float('inf')
            
            boundaries.append(boundary)
            cumulative_alpha = alpha_spent
        
        return np.array(boundaries)
    
    def _pocock_boundaries(self, alpha: float, K: int) -> np.ndarray:
        """Pocock boundaries (equal alpha spending)"""
        # Solve for critical value c such that overall alpha is maintained
        def alpha_total(c):
            # For K analyses, total alpha spent
            return K * (2 * (1 - stats.norm.cdf(c)))
        
        # Find c such that total alpha = desired alpha
        try:
            c = brentq(lambda x: alpha_total(x) - alpha, 1.5, 5.0)
        except ValueError:
            # Fallback if root finding fails
            c = 2.5
        
        return np.full(K, c)
    
    def _haybittle_peto_boundaries(self, alpha: float, K: int) -> np.ndarray:
        """Haybittle-Peto boundaries (conservative early stopping)"""
        boundaries = np.full(K-1, 3.0)  # Very conservative early boundaries
        # Final analysis uses remaining alpha
        boundaries = np.append(boundaries, stats.norm.ppf(1 - alpha/2))
        return boundaries
    
    def _calculate_futility_boundaries(self) -> np.ndarray:
        """Calculate futility boundaries for early stopping due to low power"""
        beta = self.config.beta
        t = self.information_fractions
        
        # Futility boundaries based on conditional power
        # Stop if conditional power < 20% assuming minimum effect size
        futility_boundaries = []
        
        for t_i in t[:-1]:  # No futility at final analysis
            # Critical value for futility (typically around 0 to 0.5)
            futility_boundaries.append(0.0)
        
        futility_boundaries.append(None)  # No futility boundary at final analysis
        return np.array(futility_boundaries, dtype=object)
    
    def conduct_interim_analysis(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        control_covariate: Optional[np.ndarray] = None,
        treatment_covariate: Optional[np.ndarray] = None,
        analysis_time: Optional[datetime] = None
    ) -> AnalysisResult:
        """
        Conduct an interim analysis with optional CUPED adjustment.
        
        Args:
            control_data: Control group outcomes
            treatment_data: Treatment group outcomes  
            control_covariate: Control group covariates (for CUPED)
            treatment_covariate: Treatment group covariates (for CUPED)
            analysis_time: Time of analysis
            
        Returns:
            AnalysisResult with stopping recommendation
        """
        if self.experiment_stopped:
            raise ValueError("Experiment has already been stopped")
        
        analysis_number = len(self.analyses_completed) + 1
        
        # Calculate current information fraction
        current_n = len(control_data) + len(treatment_data)
        max_n = self.config.max_sample_size * 2  # Total across both groups
        information_fraction = min(current_n / max_n, 1.0)
        
        # Perform statistical test (with CUPED if covariates provided)
        if control_covariate is not None and treatment_covariate is not None:
            # Use CUPED adjustment
            test_result = self._cuped_test_statistic(
                control_data, treatment_data, 
                control_covariate, treatment_covariate
            )
        else:
            # Standard t-test
            test_result = self._standard_test_statistic(control_data, treatment_data)
        
        # Get boundaries for this analysis
        efficacy_boundary = self.efficacy_boundaries[analysis_number - 1]
        futility_boundary = self.futility_boundaries[analysis_number - 1] if self.futility_boundaries is not None else None
        
        # Make stopping decision
        stopping_decision = self._make_stopping_decision(
            test_result["test_statistic"],
            efficacy_boundary,
            futility_boundary,
            test_result["effect_estimate"],
            information_fraction,
            current_n
        )
        
        # Calculate current power
        current_power = self._calculate_conditional_power(
            test_result["effect_estimate"],
            test_result["se"],
            information_fraction
        )
        
        # Create analysis result
        result = AnalysisResult(
            analysis_number=analysis_number,
            information_fraction=information_fraction,
            test_statistic=test_result["test_statistic"],
            p_value=test_result["p_value"],
            effect_estimate=test_result["effect_estimate"],
            confidence_interval=test_result["confidence_interval"],
            stopping_decision=stopping_decision,
            efficacy_boundary=efficacy_boundary,
            futility_boundary=futility_boundary,
            sample_size_current=current_n,
            power_current=current_power,
            recommendation=self._generate_recommendation(stopping_decision, test_result, current_power)
        )
        
        # Update experiment status
        self.analyses_completed.append(result)
        if stopping_decision != StoppingDecision.CONTINUE:
            self.experiment_stopped = True
            self.stopping_reason = stopping_decision
        
        return result
    
    def _cuped_test_statistic(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        control_covariate: np.ndarray,
        treatment_covariate: np.ndarray
    ) -> Dict:
        """Calculate test statistic with CUPED adjustment"""
        # Import CUPED function (assumes it's available)
        try:
            from app.services.experimentation import calculate_cuped_adjusted_metric
            
            cuped_results = calculate_cuped_adjusted_metric(
                control_metric=control_data,
                control_covariate=control_covariate,
                treatment_metric=treatment_data,
                treatment_covariate=treatment_covariate
            )
            
            return {
                "test_statistic": cuped_results["t_stat_cuped"],
                "p_value": cuped_results["p_value_cuped"],
                "effect_estimate": cuped_results["uplift_cuped"],
                "se": cuped_results["se_cuped"],
                "confidence_interval": cuped_results["ci_cuped"]
            }
            
        except ImportError:
            # Fallback to standard test
            return self._standard_test_statistic(control_data, treatment_data)
    
    def _standard_test_statistic(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray
    ) -> Dict:
        """Calculate standard t-test statistic"""
        n_control = len(control_data)
        n_treatment = len(treatment_data)
        
        mean_control = np.mean(control_data)
        mean_treatment = np.mean(treatment_data)
        
        var_control = np.var(control_data, ddof=1)
        var_treatment = np.var(treatment_data, ddof=1)
        
        # Pooled standard error
        se = np.sqrt(var_control/n_control + var_treatment/n_treatment)
        
        effect_estimate = mean_treatment - mean_control
        test_statistic = effect_estimate / se if se > 0 else 0
        
        # Two-tailed p-value
        df = n_control + n_treatment - 2
        p_value = 2 * (1 - stats.t.cdf(abs(test_statistic), df))
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - self.config.alpha/2, df)
        ci_lower = effect_estimate - t_critical * se
        ci_upper = effect_estimate + t_critical * se
        
        return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "effect_estimate": effect_estimate,
            "se": se,
            "confidence_interval": (ci_lower, ci_upper)
        }
    
    def _make_stopping_decision(
        self,
        test_statistic: float,
        efficacy_boundary: float,
        futility_boundary: Optional[float],
        effect_estimate: float,
        information_fraction: float,
        current_n: int
    ) -> StoppingDecision:
        """Make stopping decision based on boundaries and current results"""
        
        # Check for safety stopping (if negative effect is significant)
        if self.config.safety_threshold is not None:
            if effect_estimate < -self.config.safety_threshold and abs(test_statistic) > efficacy_boundary:
                return StoppingDecision.STOP_FOR_SAFETY
        
        # Check for efficacy stopping
        if abs(test_statistic) >= efficacy_boundary:
            return StoppingDecision.STOP_FOR_EFFICACY
        
        # Check for futility stopping (only if enabled and not final analysis)
        if (futility_boundary is not None and 
            information_fraction < 1.0 and
            test_statistic < futility_boundary):
            
            # Additional futility check based on conditional power
            conditional_power = self._calculate_conditional_power(
                effect_estimate, 
                1.0,  # Placeholder SE 
                information_fraction
            )
            
            if conditional_power < 0.2:  # Less than 20% chance of success
                return StoppingDecision.STOP_FOR_FUTILITY
        
        # Check if we've reached maximum sample size
        if current_n >= self.config.max_sample_size * 2:
            return StoppingDecision.STOP_FOR_EFFICACY  # Final analysis
        
        return StoppingDecision.CONTINUE
    
    def _calculate_conditional_power(
        self,
        current_effect: float,
        current_se: float,
        information_fraction: float
    ) -> float:
        """Calculate conditional power given current results"""
        if information_fraction >= 1.0:
            return 1.0 if abs(current_effect) >= self.config.min_effect_size else 0.0
        
        # Remaining information fraction
        remaining_fraction = 1.0 - information_fraction
        
        if remaining_fraction <= 0:
            return 1.0
        
        # Expected final test statistic assuming current effect continues
        expected_final_z = current_effect / (current_se * np.sqrt(remaining_fraction))
        
        # Final critical value
        final_boundary = self.efficacy_boundaries[-1]
        
        # Conditional power
        return 1 - stats.norm.cdf(final_boundary - expected_final_z)
    
    def _generate_recommendation(
        self,
        decision: StoppingDecision,
        test_result: Dict,
        current_power: float
    ) -> str:
        """Generate human-readable recommendation"""
        effect = test_result["effect_estimate"]
        ci = test_result["confidence_interval"]
        
        if decision == StoppingDecision.STOP_FOR_EFFICACY:
            if effect > 0:
                return f"STOP - Significant positive effect detected: {effect:.3f} (CI: {ci[0]:.3f}, {ci[1]:.3f})"
            else:
                return f"STOP - Significant negative effect detected: {effect:.3f} (CI: {ci[0]:.3f}, {ci[1]:.3f})"
        
        elif decision == StoppingDecision.STOP_FOR_FUTILITY:
            return f"STOP - Futility: Low probability of detecting meaningful effect (Power: {current_power:.1%})"
        
        elif decision == StoppingDecision.STOP_FOR_SAFETY:
            return f"STOP - Safety: Significant harmful effect detected: {effect:.3f}"
        
        else:
            return f"CONTINUE - Current effect: {effect:.3f} (CI: {ci[0]:.3f}, {ci[1]:.3f}), Power: {current_power:.1%}"
    
    def get_experiment_status(self) -> Dict:
        """Get current experiment status summary"""
        if not self.analyses_completed:
            return {
                "status": "not_started",
                "analyses_completed": 0,
                "experiment_stopped": False,
                "next_analysis_at": None
            }
        
        latest_analysis = self.analyses_completed[-1]
        
        return {
            "status": "stopped" if self.experiment_stopped else "running",
            "analyses_completed": len(self.analyses_completed),
            "experiment_stopped": self.experiment_stopped,
            "stopping_reason": self.stopping_reason.value if self.stopping_reason else None,
            "latest_analysis": {
                "analysis_number": latest_analysis.analysis_number,
                "information_fraction": latest_analysis.information_fraction,
                "effect_estimate": latest_analysis.effect_estimate,
                "p_value": latest_analysis.p_value,
                "recommendation": latest_analysis.recommendation,
                "power_current": latest_analysis.power_current
            },
            "next_analysis_at": self._estimate_next_analysis_time() if not self.experiment_stopped else None
        }
    
    def _estimate_next_analysis_time(self) -> Optional[datetime]:
        """Estimate when next analysis should be conducted"""
        if self.experiment_stopped or len(self.analyses_completed) >= self.config.max_analyses:
            return None
        
        # Simple estimation based on equal time intervals
        # In practice, this would be based on traffic patterns
        days_between_analyses = 7  # Weekly analyses
        return datetime.now() + timedelta(days=days_between_analyses)
    
    def plot_boundaries(self) -> None:
        """Plot efficacy and futility boundaries"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot efficacy boundaries
        ax.plot(self.information_fractions, self.efficacy_boundaries, 
                'r-', marker='o', label='Efficacy Boundary (Upper)', linewidth=2)
        ax.plot(self.information_fractions, -self.efficacy_boundaries, 
                'r--', marker='o', label='Efficacy Boundary (Lower)', linewidth=2)
        
        # Plot futility boundaries if available
        if self.futility_boundaries is not None:
            futility_valid = [b for b in self.futility_boundaries if b is not None]
            fractions_valid = self.information_fractions[:len(futility_valid)]
            if futility_valid:
                ax.plot(fractions_valid, futility_valid, 
                        'b-', marker='s', label='Futility Boundary', linewidth=2)
        
        # Plot completed analyses
        if self.analyses_completed:
            fractions = [a.information_fraction for a in self.analyses_completed]
            test_stats = [a.test_statistic for a in self.analyses_completed]
            
            # Different colors based on stopping decision
            colors = []
            for analysis in self.analyses_completed:
                if analysis.stopping_decision == StoppingDecision.STOP_FOR_EFFICACY:
                    colors.append('green')
                elif analysis.stopping_decision == StoppingDecision.STOP_FOR_FUTILITY:
                    colors.append('orange')
                elif analysis.stopping_decision == StoppingDecision.STOP_FOR_SAFETY:
                    colors.append('red')
                else:
                    colors.append('blue')
            
            ax.scatter(fractions, test_stats, c=colors, s=100, zorder=5, 
                      label='Analyses Completed', edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Information Fraction')
        ax.set_ylabel('Test Statistic (Z-score)')
        ax.set_title(f'Sequential Monitoring Boundaries ({self.config.spending_function})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

# Utility functions for common sequential testing scenarios
def create_sequential_experiment(
    experiment_id: str,
    target_effect: float,
    baseline_rate: float,
    max_sample_size: int,
    power: float = 0.8,
    alpha: float = 0.05,
    spending_function: str = "obrien_fleming"
) -> SequentialMonitor:
    """Create a sequential experiment with reasonable defaults"""
    
    config = ExperimentConfig(
        experiment_id=experiment_id,
        alpha=alpha,
        beta=1-power,
        max_sample_size=max_sample_size,
        min_effect_size=abs(target_effect),
        spending_function=spending_function,
        futility_enabled=True
    )
    
    return SequentialMonitor(config)

def run_sequential_simulation(
    true_effect: float,
    sample_size_per_analysis: int,
    max_analyses: int = 5,
    baseline_std: float = 1.0,
    spending_function: str = "obrien_fleming"
) -> Dict:
    """Simulate a sequential experiment to test the monitoring system"""
    
    # Create monitor
    config = ExperimentConfig(
        experiment_id="simulation",
        max_analyses=max_analyses,
        max_sample_size=sample_size_per_analysis * max_analyses,
        spending_function=spending_function
    )
    
    monitor = SequentialMonitor(config)
    
    # Generate data progressively
    np.random.seed(42)
    results = []
    
    for analysis in range(1, max_analyses + 1):
        # Cumulative sample size
        current_n = analysis * sample_size_per_analysis
        
        # Generate control data
        control_data = np.random.normal(0, baseline_std, current_n)
        
        # Generate treatment data with true effect
        treatment_data = np.random.normal(true_effect, baseline_std, current_n)
        
        # Conduct analysis
        result = monitor.conduct_interim_analysis(control_data, treatment_data)
        results.append(result)
        
        # Stop if recommended
        if result.stopping_decision != StoppingDecision.CONTINUE:
            break
    
    return {
        "results": results,
        "stopped_early": monitor.experiment_stopped,
        "stopping_reason": monitor.stopping_reason,
        "final_analysis": results[-1] if results else None
    }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Sequential Monitoring System...")
    
    # Test 1: Create a sequential monitor
    monitor = create_sequential_experiment(
        experiment_id="test_conversion",
        target_effect=0.01,  # 1 percentage point increase
        baseline_rate=0.05,  # 5% baseline
        max_sample_size=10000,
        spending_function="obrien_fleming"
    )
    
    print(f"Created sequential monitor with {monitor.config.max_analyses} planned analyses")
    print(f"   Efficacy boundaries: {monitor.efficacy_boundaries}")
    
    # Test 2: Simulate experiment with positive effect
    print(f"\n Simulating experiment with 2% positive effect...")
    sim_results = run_sequential_simulation(
        true_effect=0.02,
        sample_size_per_analysis=1000,
        max_analyses=5
    )
    
    print(f"   Stopped early: {sim_results['stopped_early']}")
    if sim_results['final_analysis']:
        final = sim_results['final_analysis']
        print(f"   Final decision: {final.stopping_decision.value}")
        print(f"   Effect estimate: {final.effect_estimate:.4f}")
        print(f"   Analysis number: {final.analysis_number}")
    
    # Test 3: Simulate A/A test (should not stop early for efficacy)
    print(f"\n Simulating A/A test (no effect)...")
    aa_results = run_sequential_simulation(
        true_effect=0.00,
        sample_size_per_analysis=1000,
        max_analyses=5
    )
    
    print(f"   Stopped early: {aa_results['stopped_early']}")
    if aa_results['final_analysis']:
        final = aa_results['final_analysis']
        print(f"   Final decision: {final.stopping_decision.value}")
        print(f"   Effect estimate: {final.effect_estimate:.4f}")
    
    print(f"\n Sequential monitoring system tests completed!")