import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings

class MDECalculator:
    """
    Minimum Detectable Effect Calculator with CUPED integration.
    Supports both continuous and binary metrics.
    """
    
    def __init__(self):
        self.default_alpha = 0.05
        self.default_power = 0.8
        self.default_ratio = 1.0  # 1:1 allocation
    
    def calculate_mde_continuous(
        self,
        sample_size_per_group: int,
        pooled_std: float,
        alpha: float = None,
        power: float = None,
        allocation_ratio: float = None,
        variance_reduction: float = 0.0,
        two_sided: bool = True
    ) -> Dict:
        """
        Calculate MDE for continuous metrics.
        
        Args:
            sample_size_per_group: Sample size per group
            pooled_std: Pooled standard deviation of the metric
            alpha: Significance level (default 0.05)
            power: Statistical power (default 0.8)
            allocation_ratio: Treatment/Control allocation ratio (default 1.0)
            variance_reduction: Variance reduction from CUPED (0-1)
            two_sided: Whether to use two-sided test
            
        Returns:
            Dictionary with MDE and related statistics
        """
        alpha = alpha or self.default_alpha
        power = power or self.default_power
        allocation_ratio = allocation_ratio or self.default_ratio
        
        # Validate inputs
        if sample_size_per_group <= 0:
            raise ValueError("Sample size must be positive")
        if pooled_std <= 0:
            raise ValueError("Standard deviation must be positive")
        if not 0 <= variance_reduction < 1:
            raise ValueError("Variance reduction must be between 0 and 1")
        
        # Adjust standard deviation for CUPED
        adjusted_std = pooled_std * np.sqrt(1 - variance_reduction)
        
        # Calculate critical values
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Calculate effective sample sizes
        n_control = sample_size_per_group
        n_treatment = int(sample_size_per_group * allocation_ratio)
        
        # Standard error for difference in means
        se_diff = adjusted_std * np.sqrt(1/n_control + 1/n_treatment)
        
        # Calculate MDE
        mde = (z_alpha + z_beta) * se_diff
        
        # Relative MDE (as percentage of control mean, if provided)
        relative_mde = None
        
        return {
            "mde_absolute": mde,
            "mde_relative": relative_mde,
            "sample_size_control": n_control,
            "sample_size_treatment": n_treatment,
            "total_sample_size": n_control + n_treatment,
            "pooled_std_original": pooled_std,
            "pooled_std_adjusted": adjusted_std,
            "variance_reduction": variance_reduction,
            "power": power,
            "alpha": alpha,
            "two_sided": two_sided,
            "allocation_ratio": allocation_ratio,
            "se_difference": se_diff,
            "z_alpha": z_alpha,
            "z_beta": z_beta
        }
    
    def calculate_mde_binary(
        self,
        sample_size_per_group: int,
        baseline_rate: float,
        alpha: float = None,
        power: float = None,
        allocation_ratio: float = None,
        variance_reduction: float = 0.0,
        two_sided: bool = True
    ) -> Dict:
        """
        Calculate MDE for binary metrics (conversion rates, etc.).
        
        Args:
            sample_size_per_group: Sample size per group
            baseline_rate: Baseline conversion rate (0-1)
            alpha: Significance level
            power: Statistical power
            allocation_ratio: Treatment/Control allocation ratio
            variance_reduction: Variance reduction from CUPED
            two_sided: Whether to use two-sided test
            
        Returns:
            Dictionary with MDE and related statistics
        """
        alpha = alpha or self.default_alpha
        power = power or self.default_power
        allocation_ratio = allocation_ratio or self.default_ratio
        
        # Validate inputs
        if not 0 < baseline_rate < 1:
            raise ValueError("Baseline rate must be between 0 and 1")
        
        # Calculate pooled standard deviation for binary metric
        pooled_variance = baseline_rate * (1 - baseline_rate)
        pooled_std = np.sqrt(pooled_variance)
        
        # Use continuous formula with binary-specific adjustments
        result = self.calculate_mde_continuous(
            sample_size_per_group=sample_size_per_group,
            pooled_std=pooled_std,
            alpha=alpha,
            power=power,
            allocation_ratio=allocation_ratio,
            variance_reduction=variance_reduction,
            two_sided=two_sided
        )
        
        # Add binary-specific metrics
        mde_absolute = result["mde_absolute"]
        mde_relative = mde_absolute / baseline_rate
        
        result.update({
            "baseline_rate": baseline_rate,
            "mde_relative": mde_relative,
            "mde_percentage_points": mde_absolute,
            "treatment_rate_lower": baseline_rate - mde_absolute,
            "treatment_rate_upper": baseline_rate + mde_absolute,
            "metric_type": "binary"
        })
        
        return result
    
    def calculate_sample_size_continuous(
        self,
        mde: float,
        pooled_std: float,
        alpha: float = None,
        power: float = None,
        allocation_ratio: float = None,
        variance_reduction: float = 0.0,
        two_sided: bool = True
    ) -> Dict:
        """
        Calculate required sample size for continuous metrics.
        
        Args:
            mde: Minimum detectable effect (absolute)
            pooled_std: Pooled standard deviation
            alpha: Significance level
            power: Statistical power
            allocation_ratio: Treatment/Control allocation ratio
            variance_reduction: Variance reduction from CUPED
            two_sided: Whether to use two-sided test
            
        Returns:
            Dictionary with sample size requirements
        """
        alpha = alpha or self.default_alpha
        power = power or self.default_power
        allocation_ratio = allocation_ratio or self.default_ratio
        
        # Validate inputs
        if mde <= 0:
            raise ValueError("MDE must be positive")
        if pooled_std <= 0:
            raise ValueError("Standard deviation must be positive")
        
        # Adjust standard deviation for CUPED
        adjusted_std = pooled_std * np.sqrt(1 - variance_reduction)
        
        # Calculate critical values
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Calculate required sample size per group (control)
        # Formula: n = (Z_α + Z_β)² × σ² × (1 + 1/r) / δ²
        # where r = allocation_ratio
        
        variance_multiplier = (1 + 1/allocation_ratio)
        n_control = ((z_alpha + z_beta) ** 2 * adjusted_std ** 2 * variance_multiplier) / (mde ** 2)
        n_control = int(np.ceil(n_control))
        
        n_treatment = int(np.ceil(n_control * allocation_ratio))
        
        return {
            "sample_size_control": n_control,
            "sample_size_treatment": n_treatment,
            "total_sample_size": n_control + n_treatment,
            "mde_target": mde,
            "pooled_std_original": pooled_std,
            "pooled_std_adjusted": adjusted_std,
            "variance_reduction": variance_reduction,
            "power": power,
            "alpha": alpha,
            "two_sided": two_sided,
            "allocation_ratio": allocation_ratio,
            "z_alpha": z_alpha,
            "z_beta": z_beta,
            "variance_multiplier": variance_multiplier
        }
    
    def calculate_sample_size_binary(
        self,
        mde_relative: float,
        baseline_rate: float,
        alpha: float = None,
        power: float = None,
        allocation_ratio: float = None,
        variance_reduction: float = 0.0,
        two_sided: bool = True
    ) -> Dict:
        """
        Calculate required sample size for binary metrics.
        
        Args:
            mde_relative: Minimum detectable effect as relative change (e.g., 0.1 for 10% relative increase)
            baseline_rate: Baseline conversion rate (0-1)
            alpha: Significance level
            power: Statistical power
            allocation_ratio: Treatment/Control allocation ratio
            variance_reduction: Variance reduction from CUPED
            two_sided: Whether to use two-sided test
            
        Returns:
            Dictionary with sample size requirements
        """
        # Convert relative MDE to absolute
        mde_absolute = baseline_rate * mde_relative
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(baseline_rate * (1 - baseline_rate))
        
        # Use continuous formula
        result = self.calculate_sample_size_continuous(
            mde=mde_absolute,
            pooled_std=pooled_std,
            alpha=alpha,
            power=power,
            allocation_ratio=allocation_ratio,
            variance_reduction=variance_reduction,
            two_sided=two_sided
        )
        
        # Add binary-specific info
        result.update({
            "baseline_rate": baseline_rate,
            "mde_relative": mde_relative,
            "mde_absolute": mde_absolute,
            "treatment_rate_target": baseline_rate * (1 + mde_relative),
            "metric_type": "binary"
        })
        
        return result
    
    def power_analysis(
        self,
        effect_sizes: List[float],
        sample_sizes: List[int],
        pooled_std: float,
        alpha: float = None,
        allocation_ratio: float = None,
        variance_reduction: float = 0.0,
        two_sided: bool = True
    ) -> pd.DataFrame:
        """
        Perform power analysis across different effect sizes and sample sizes.
        
        Args:
            effect_sizes: List of effect sizes to test
            sample_sizes: List of sample sizes to test
            pooled_std: Pooled standard deviation
            alpha: Significance level
            allocation_ratio: Treatment/Control allocation ratio
            variance_reduction: Variance reduction from CUPED
            two_sided: Whether to use two-sided test
            
        Returns:
            DataFrame with power analysis results
        """
        alpha = alpha or self.default_alpha
        allocation_ratio = allocation_ratio or self.default_ratio
        
        results = []
        
        for effect_size in effect_sizes:
            for sample_size in sample_sizes:
                # Calculate power for this combination
                mde_result = self.calculate_mde_continuous(
                    sample_size_per_group=sample_size,
                    pooled_std=pooled_std,
                    alpha=alpha,
                    power=0.8,  # This will be overridden
                    allocation_ratio=allocation_ratio,
                    variance_reduction=variance_reduction,
                    two_sided=two_sided
                )
                
                # Calculate actual power for this effect size
                se_diff = mde_result["se_difference"]
                z_alpha = mde_result["z_alpha"]
                
                # Power = P(Z > z_α - δ/SE)
                z_power = z_alpha - (effect_size / se_diff)
                power = 1 - stats.norm.cdf(z_power)
                
                results.append({
                    "effect_size": effect_size,
                    "sample_size_per_group": sample_size,
                    "total_sample_size": sample_size * (1 + allocation_ratio),
                    "power": power,
                    "mde": mde_result["mde_absolute"],
                    "se_difference": se_diff,
                    "variance_reduction": variance_reduction
                })
        
        return pd.DataFrame(results)
    
    def cuped_benefit_analysis(
        self,
        correlations: List[float],
        sample_size: int,
        effect_size: float,
        pooled_std: float,
        alpha: float = None
    ) -> pd.DataFrame:
        """
        Analyze the benefit of CUPED across different correlation levels.
        
        Args:
            correlations: List of correlations between covariate and outcome
            sample_size: Sample size per group
            effect_size: True effect size
            pooled_std: Pooled standard deviation
            alpha: Significance level
            
        Returns:
            DataFrame with CUPED benefit analysis
        """
        alpha = alpha or self.default_alpha
        
        results = []
        
        for corr in correlations:
            variance_reduction = corr ** 2
            
            # Without CUPED
            no_cuped = self.calculate_mde_continuous(
                sample_size_per_group=sample_size,
                pooled_std=pooled_std,
                alpha=alpha,
                variance_reduction=0.0
            )
            
            # With CUPED
            with_cuped = self.calculate_mde_continuous(
                sample_size_per_group=sample_size,
                pooled_std=pooled_std,
                alpha=alpha,
                variance_reduction=variance_reduction
            )
            
            # Calculate power for detecting the true effect
            se_no_cuped = no_cuped["se_difference"]
            se_cuped = with_cuped["se_difference"]
            z_alpha = no_cuped["z_alpha"]
            
            power_no_cuped = 1 - stats.norm.cdf(z_alpha - effect_size/se_no_cuped)
            power_cuped = 1 - stats.norm.cdf(z_alpha - effect_size/se_cuped)
            
            # Sample size reduction
            sample_size_equivalent = self.calculate_sample_size_continuous(
                mde=effect_size,
                pooled_std=pooled_std,
                alpha=alpha,
                variance_reduction=variance_reduction
            )["sample_size_control"]
            
            sample_size_reduction = (sample_size - sample_size_equivalent) / sample_size
            
            results.append({
                "correlation": corr,
                "variance_reduction": variance_reduction,
                "mde_no_cuped": no_cuped["mde_absolute"],
                "mde_cuped": with_cuped["mde_absolute"],
                "mde_improvement": (no_cuped["mde_absolute"] - with_cuped["mde_absolute"]) / no_cuped["mde_absolute"],
                "power_no_cuped": power_no_cuped,
                "power_cuped": power_cuped,
                "power_improvement": power_cuped - power_no_cuped,
                "sample_size_reduction": sample_size_reduction,
                "relative_efficiency": (se_no_cuped / se_cuped) ** 2
            })
        
        return pd.DataFrame(results)
    
    def experiment_planning_report(
        self,
        baseline_rate: float,
        target_uplift: float,
        pooled_std: Optional[float] = None,
        covariate_correlation: float = 0.0,
        alpha: float = None,
        power: float = None,
        max_sample_size: int = None,
        is_binary: bool = True
    ) -> Dict:
        """
        Generate comprehensive experiment planning report.
        
        Args:
            baseline_rate: Baseline conversion rate (for binary) or mean (for continuous)
            target_uplift: Target relative uplift (e.g., 0.05 for 5% increase)
            pooled_std: Pooled standard deviation (for continuous metrics)
            covariate_correlation: Correlation between covariate and outcome
            alpha: Significance level
            power: Statistical power
            max_sample_size: Maximum feasible sample size per group
            is_binary: Whether metric is binary
            
        Returns:
            Comprehensive planning report
        """
        alpha = alpha or self.default_alpha
        power = power or self.default_power
        variance_reduction = covariate_correlation ** 2
        
        report = {
            "experiment_setup": {
                "metric_type": "binary" if is_binary else "continuous",
                "baseline_rate": baseline_rate,
                "target_uplift": target_uplift,
                "target_absolute_effect": baseline_rate * target_uplift if is_binary else target_uplift,
                "alpha": alpha,
                "power": power,
                "covariate_correlation": covariate_correlation,
                "variance_reduction": variance_reduction
            }
        }
        
        # Calculate sample size requirements
        if is_binary:
            sample_result = self.calculate_sample_size_binary(
                mde_relative=target_uplift,
                baseline_rate=baseline_rate,
                alpha=alpha,
                power=power,
                variance_reduction=variance_reduction
            )
        else:
            if pooled_std is None:
                raise ValueError("pooled_std is required for continuous metrics")
            sample_result = self.calculate_sample_size_continuous(
                mde=target_uplift,
                pooled_std=pooled_std,
                alpha=alpha,
                power=power,
                variance_reduction=variance_reduction
            )
        
        report["sample_size_requirements"] = sample_result
        
        # Check feasibility
        if max_sample_size is not None:
            feasible = sample_result["sample_size_control"] <= max_sample_size
            report["feasibility"] = {
                "max_feasible_sample_size": max_sample_size,
                "required_sample_size": sample_result["sample_size_control"],
                "is_feasible": feasible
            }
            
            if not feasible:
                # Calculate what's achievable with max sample size
                if is_binary:
                    achievable_mde = self.calculate_mde_binary(
                        sample_size_per_group=max_sample_size,
                        baseline_rate=baseline_rate,
                        alpha=alpha,
                        power=power,
                        variance_reduction=variance_reduction
                    )
                else:
                    achievable_mde = self.calculate_mde_continuous(
                        sample_size_per_group=max_sample_size,
                        pooled_std=pooled_std,
                        alpha=alpha,
                        power=power,
                        variance_reduction=variance_reduction
                    )
                
                report["feasibility"]["achievable_mde"] = achievable_mde
        
        # CUPED benefit analysis
        if covariate_correlation > 0:
            cuped_benefit = self.cuped_benefit_analysis(
                correlations=[0, covariate_correlation],
                sample_size=sample_result["sample_size_control"],
                effect_size=baseline_rate * target_uplift if is_binary else target_uplift,
                pooled_std=np.sqrt(baseline_rate * (1 - baseline_rate)) if is_binary else pooled_std,
                alpha=alpha
            )
            report["cuped_benefit"] = cubed_benefit.to_dict('records')
        
        return report


# Utility functions for common use cases
def quick_sample_size(baseline_rate: float, target_uplift: float, power: float = 0.8) -> int:
    """Quick sample size calculation for binary metrics."""
    calc = MDECalculator()
    result = calc.calculate_sample_size_binary(
        mde_relative=target_uplift,
        baseline_rate=baseline_rate,
        power=power
    )
    return result["sample_size_control"]

def quick_mde(sample_size: int, baseline_rate: float, power: float = 0.8) -> float:
    """Quick MDE calculation for binary metrics."""
    calc = MDECalculator()
    result = calc.calculate_mde_binary(
        sample_size_per_group=sample_size,
        baseline_rate=baseline_rate,
        power=power
    )
    return result["mde_relative"]

# Example usage
if __name__ == "__main__":
    calc = MDECalculator()
    
    # Example 1: Sample size for conversion rate experiment
    print("=== Conversion Rate Experiment Planning ===")
    sample_result = calc.calculate_sample_size_binary(
        mde_relative=0.05,  # 5% relative increase
        baseline_rate=0.10,  # 10% baseline conversion
        power=0.8,
        variance_reduction=0.25  # 25% variance reduction from CUPED
    )
    print(f"Required sample size per group: {sample_result['sample_size_control']:,}")
    print(f"Total sample size: {sample_result['total_sample_size']:,}")
    print(f"Target conversion rate: {sample_result['treatment_rate_target']:.1%}")
    
    # Example 2: MDE for fixed sample size
    print("\n=== MDE for Fixed Sample Size ===")
    mde_result = calc.calculate_mde_binary(
        sample_size_per_group=5000,
        baseline_rate=0.10,
        variance_reduction=0.25
    )
    print(f"Detectable relative uplift: {mde_result['mde_relative']:.1%}")
    print(f"Detectable absolute uplift: {mde_result['mde_percentage_points']:.1%} pts")
    
    # Example 3: Power analysis
    print("\n=== Power Analysis ===")
    power_df = calc.power_analysis(
        effect_sizes=[0.005, 0.01, 0.015, 0.02],  # Absolute effect sizes
        sample_sizes=[1000, 2000, 5000, 10000],
        pooled_std=0.3,  # For continuous metric
        variance_reduction=0.2
    )
    print(power_df.head())