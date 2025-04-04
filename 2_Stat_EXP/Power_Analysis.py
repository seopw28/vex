#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Power Analysis for A/B Testing
This script provides functions for power analysis for CTR and Revenue Amount metrics.
"""

#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from statsmodels.stats.power import TTestPower
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.power import tt_ind_solve_power

#%% CTR Power Analysis Function
def calculate_ctr_power_analysis(
    baseline_ctr=0.02,
    minimum_detectable_effect=0.002,  # 0.1% absolute change
    alpha=0.05,
    power=0.8,
    traffic_split=0.5
):
    """
    Calculate required sample size for CTR power analysis.
    
    Parameters:
    -----------
    baseline_ctr : float
        The baseline click-through rate (e.g., 0.02 for 2%)
    minimum_detectable_effect : float
        The minimum absolute change in CTR to detect (e.g., 0.002 for 0.2%)
    alpha : float
        Significance level (default: 0.05)
    power : float
        Desired statistical power (default: 0.8)
    traffic_split : float
        Proportion of traffic to allocate to each variant (default: 0.5)
        
    Returns:
    --------
    dict
        Dictionary containing power analysis results
    """
    # Calculate effect size (Cohen's h for proportions)
    p1 = baseline_ctr
    p2 = baseline_ctr + minimum_detectable_effect
    
    # Cohen's h calculation for proportions
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    
    # Calculate required sample size
    n = tt_ind_solve_power(
        effect_size=h,
        alpha=alpha,
        power=power,
        ratio=traffic_split/(1-traffic_split),
        alternative='two-sided'
    )
    
    # Round up to nearest integer
    n = int(np.ceil(n))
    
    # Calculate total sample size needed
    total_n = n * (1/traffic_split)
    
    # Calculate actual power with rounded sample size
    actual_power = tt_ind_solve_power(
        effect_size=h,
        nobs1=n,
        alpha=alpha,
        ratio=traffic_split/(1-traffic_split),
        alternative='two-sided'
    )
    
    return {
        'baseline_ctr': baseline_ctr,
        'minimum_detectable_effect': minimum_detectable_effect,
        'effect_size': h,
        'sample_size_per_variant': n,
        'total_sample_size': total_n,
        'alpha': alpha,
        'power': power,
        'actual_power': actual_power,
        'traffic_split': traffic_split
    }

#%% Revenue Power Analysis Function
def calculate_revenue_power_analysis(
    baseline_mean=50.0,
    baseline_std=25.0,
    minimum_detectable_effect=5.0,  # $5 absolute change
    alpha=0.05,
    power=0.8,
    traffic_split=0.5
):
    """
    Calculate required sample size for Revenue Amount power analysis.
    
    Parameters:
    -----------
    baseline_mean : float
        The baseline mean revenue per user
    baseline_std : float
        The baseline standard deviation of revenue per user
    minimum_detectable_effect : float
        The minimum absolute change in revenue to detect
    alpha : float
        Significance level (default: 0.05)
    power : float
        Desired statistical power (default: 0.8)
    traffic_split : float
        Proportion of traffic to allocate to each variant (default: 0.5)
        
    Returns:
    --------
    dict
        Dictionary containing power analysis results
    """
    # Calculate effect size (Cohen's d)
    d = minimum_detectable_effect / baseline_std
    
    # Calculate required sample size
    n = tt_ind_solve_power(
        effect_size=d,
        alpha=alpha,
        power=power,
        ratio=traffic_split/(1-traffic_split),
        alternative='two-sided'
    )
    
    # Round up to nearest integer
    n = int(np.ceil(n))
    
    # Calculate total sample size needed
    total_n = n * (1/traffic_split)
    
    # Calculate actual power with rounded sample size
    actual_power = tt_ind_solve_power(
        effect_size=d,
        nobs1=n,
        alpha=alpha,
        ratio=traffic_split/(1-traffic_split),
        alternative='two-sided'
    )
    
    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'minimum_detectable_effect': minimum_detectable_effect,
        'effect_size': d,
        'sample_size_per_variant': n,
        'total_sample_size': total_n,
        'alpha': alpha,
        'power': power,
        'actual_power': actual_power,
        'traffic_split': traffic_split
    }

#%% Power Curve Visualization Function
def plot_power_curve(metric_type='ctr', **kwargs):
    """
    Plot power curves for different sample sizes.
    
    Parameters:
    -----------
    metric_type : str
        Either 'ctr' or 'revenue'
    **kwargs : dict
        Parameters for the power analysis functions
    """
    if metric_type.lower() == 'ctr':
        baseline = kwargs.get('baseline_ctr', 0.02)
        mde = kwargs.get('minimum_detectable_effect', 0.002)
        p1 = baseline
        p2 = baseline + mde
        h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
        effect_size = h
        title = f"Power Curve for CTR (Baseline: {baseline:.3f}, MDE: {mde:.3f})"
    else:  # revenue
        baseline = kwargs.get('baseline_mean', 50.0)
        std = kwargs.get('baseline_std', 25.0)
        mde = kwargs.get('minimum_detectable_effect', 5.0)
        d = mde / std
        effect_size = d
        title = f"Power Curve for Revenue (Baseline: ${baseline:.2f}, MDE: ${mde:.2f})"
    
    # Generate sample sizes
    sample_sizes = np.linspace(100, 10000, 100)
    powers = []
    
    for n in sample_sizes:
        n = int(n)
        power = tt_ind_solve_power(
            effect_size=effect_size,
            nobs1=n,
            alpha=kwargs.get('alpha', 0.05),
            ratio=kwargs.get('traffic_split', 0.5)/(1-kwargs.get('traffic_split', 0.5)),
            alternative='two-sided'
        )
        powers.append(power)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, powers, 'b-', linewidth=2)
    plt.axhline(y=kwargs.get('power', 0.8), color='r', linestyle='--', label=f"Target Power: {kwargs.get('power', 0.8)}")
    plt.axvline(x=kwargs.get('sample_size_per_variant', 0), color='g', linestyle='--', 
                label=f"Required Sample Size: {kwargs.get('sample_size_per_variant', 0)}")
    plt.grid(True, alpha=0.3)
    plt.xlabel('Sample Size per Variant')
    plt.ylabel('Statistical Power')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Example: Run CTR Power Analysis
# CTR Power Analysis
ctr_results = calculate_ctr_power_analysis(
    baseline_ctr=0.02,
    minimum_detectable_effect=0.002,
    alpha=0.05,
    power=0.8,
    traffic_split=0.5
)

print("=" * 50)
print("CTR POWER ANALYSIS RESULTS")
print("=" * 50)
for key, value in ctr_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

#%% Example: Run Revenue Power Analysis
# Revenue Power Analysis
revenue_results = calculate_revenue_power_analysis(
    baseline_mean=50.0,
    baseline_std=25.0,
    minimum_detectable_effect=5.0,
    alpha=0.05,
    power=0.8,
    traffic_split=0.5
)

print("\n" + "=" * 50)
print("REVENUE POWER ANALYSIS RESULTS")
print("=" * 50)
for key, value in revenue_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

#%% Example: Plot Power Curves
# Plot power curves
plot_power_curve('ctr', **ctr_results)
plot_power_curve('revenue', **revenue_results)

#%% Example: Custom CTR Analysis
# Example with different parameters
custom_ctr_results = calculate_ctr_power_analysis(
    baseline_ctr=0.05,  # 5% baseline CTR
    minimum_detectable_effect=0.005,  # 0.5% absolute change
    alpha=0.01,  # 1% significance level
    power=0.9,  # 90% statistical power
    traffic_split=0.7  # 70/30 split
)

print("=" * 50)
print("CUSTOM CTR POWER ANALYSIS RESULTS")
print("=" * 50)
for key, value in custom_ctr_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

#%% Example: Custom Revenue Analysis
# Example with different parameters
custom_revenue_results = calculate_revenue_power_analysis(
    baseline_mean=100.0,  # $100 baseline revenue
    baseline_std=50.0,  # $50 standard deviation
    minimum_detectable_effect=10.0,  # $10 absolute change
    alpha=0.01,  # 1% significance level
    power=0.9,  # 90% statistical power
    traffic_split=0.7  # 70/30 split
)

print("=" * 50)
print("CUSTOM REVENUE POWER ANALYSIS RESULTS")
print("=" * 50)
for key, value in custom_revenue_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

#%% Example: Plot Custom Power Curves
# Plot custom power curves
plot_power_curve('ctr', **custom_ctr_results)
plot_power_curve('revenue', **custom_revenue_results)

#%% Example: Compare Different Traffic Splits
# Compare different traffic splits for CTR
traffic_splits = [0.5, 0.6, 0.7, 0.8, 0.9]
ctr_split_results = []

for split in traffic_splits:
    result = calculate_ctr_power_analysis(
        baseline_ctr=0.02,
        minimum_detectable_effect=0.002,
        alpha=0.05,
        power=0.8,
        traffic_split=split
    )
    ctr_split_results.append(result)

# Create a DataFrame for comparison
split_comparison = pd.DataFrame([
    {
        'traffic_split': result['traffic_split'],
        'sample_size_per_variant': result['sample_size_per_variant'],
        'total_sample_size': result['total_sample_size'],
        'actual_power': result['actual_power']
    }
    for result in ctr_split_results
])

print("=" * 50)
print("TRAFFIC SPLIT COMPARISON FOR CTR")
print("=" * 50)
print(split_comparison)

# Plot the comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(split_comparison['traffic_split'], split_comparison['total_sample_size'], 'o-', linewidth=2)
plt.xlabel('Traffic Split (Treatment)')
plt.ylabel('Total Sample Size Required')
plt.title('Total Sample Size vs Traffic Split')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(split_comparison['traffic_split'], split_comparison['actual_power'], 'o-', linewidth=2)
plt.xlabel('Traffic Split (Treatment)')
plt.ylabel('Actual Power')
plt.title('Actual Power vs Traffic Split')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% Example: Compare Different Effect Sizes
# Compare different effect sizes for Revenue
effect_sizes = [2.0, 5.0, 10.0, 15.0, 20.0]
revenue_effect_results = []

for effect in effect_sizes:
    result = calculate_revenue_power_analysis(
        baseline_mean=50.0,
        baseline_std=25.0,
        minimum_detectable_effect=effect,
        alpha=0.05,
        power=0.8,
        traffic_split=0.5
    )
    revenue_effect_results.append(result)

# Create a DataFrame for comparison
effect_comparison = pd.DataFrame([
    {
        'minimum_detectable_effect': result['minimum_detectable_effect'],
        'effect_size': result['effect_size'],
        'sample_size_per_variant': result['sample_size_per_variant'],
        'total_sample_size': result['total_sample_size'],
        'actual_power': result['actual_power']
    }
    for result in revenue_effect_results
])

print("=" * 50)
print("EFFECT SIZE COMPARISON FOR REVENUE")
print("=" * 50)
print(effect_comparison)

# Plot the comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(effect_comparison['minimum_detectable_effect'], effect_comparison['total_sample_size'], 'o-', linewidth=2)
plt.xlabel('Minimum Detectable Effect ($)')
plt.ylabel('Total Sample Size Required')
plt.title('Total Sample Size vs Effect Size')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(effect_comparison['minimum_detectable_effect'], effect_comparison['effect_size'], 'o-', linewidth=2)
plt.xlabel('Minimum Detectable Effect ($)')
plt.ylabel('Effect Size (Cohen\'s d)')
plt.title('Effect Size vs Minimum Detectable Effect')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% Main function (for running as a script)
def run_power_analysis_example():
    """Run example power analyses for both CTR and Revenue metrics."""
    # CTR Power Analysis
    ctr_results = calculate_ctr_power_analysis(
        baseline_ctr=0.02,
        minimum_detectable_effect=0.002,
        alpha=0.05,
        power=0.8,
        traffic_split=0.5
    )
    
    print("=" * 50)
    print("CTR POWER ANALYSIS RESULTS")
    print("=" * 50)
    for key, value in ctr_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Revenue Power Analysis
    revenue_results = calculate_revenue_power_analysis(
        baseline_mean=50.0,
        baseline_std=25.0,
        minimum_detectable_effect=5.0,
        alpha=0.05,
        power=0.8,
        traffic_split=0.5
    )
    
    print("\n" + "=" * 50)
    print("REVENUE POWER ANALYSIS RESULTS")
    print("=" * 50)
    for key, value in revenue_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Plot power curves
    plot_power_curve('ctr', **ctr_results)
    plot_power_curve('revenue', **revenue_results)

if __name__ == "__main__":
    run_power_analysis_example()
