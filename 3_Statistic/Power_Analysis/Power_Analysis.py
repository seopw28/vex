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
import io
from fpdf import FPDF
import os
import tempfile
from datetime import datetime

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

#%% PDF Report Generation Function
def generate_power_analysis_pdf(ctr_results, revenue_results, custom_ctr_results=None, custom_revenue_results=None, 
                              traffic_split_comparison=None, effect_size_comparison=None, output_path="power_analysis_report.pdf"):
    """
    Generate a comprehensive PDF report with power analysis results and visualizations.
    
    Parameters:
    -----------
    ctr_results : dict
        Results from the CTR power analysis
    revenue_results : dict
        Results from the Revenue power analysis
    custom_ctr_results : dict, optional
        Results from a custom CTR power analysis
    custom_revenue_results : dict, optional
        Results from a custom Revenue power analysis
    traffic_split_comparison : DataFrame, optional
        Comparison of different traffic splits
    effect_size_comparison : DataFrame, optional
        Comparison of different effect sizes
    output_path : str
        Path to save the PDF report
    """
    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, "Power Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    # Date and time
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="R")
    pdf.ln(10)
    
    # Introduction
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Introduction", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This report presents the results of power analysis for A/B testing. "
                "Power analysis helps determine the sample size required to detect a specified effect "
                "with a given level of confidence. The analysis is performed for two metrics: "
                "Click-Through Rate (CTR) and Revenue Amount.")
    pdf.ln(10)
    
    # CTR Power Analysis Results
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "CTR Power Analysis", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Standard Analysis", ln=True)
    pdf.set_font("Arial", "", 12)
    
    # Add CTR results table
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(80, 10, "Parameter", 1, 0, "C", True)
    pdf.cell(80, 10, "Value", 1, 1, "C", True)
    
    for key, value in ctr_results.items():
        pdf.cell(80, 10, key, 1, 0, "L")
        if isinstance(value, float):
            pdf.cell(80, 10, f"{value:.4f}", 1, 1, "R")
        else:
            pdf.cell(80, 10, str(value), 1, 1, "R")
    
    pdf.ln(10)
    
    # CTR Power Curve
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "CTR Power Curve", ln=True)
    
    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.figure(figsize=(10, 6))
        plot_power_curve('ctr', **ctr_results)
        plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add the image to the PDF
        pdf.image(tmp_file.name, x=10, y=None, w=190)
    
    # Remove the temporary file
    os.unlink(tmp_file.name)
    pdf.ln(10)
    
    # Revenue Power Analysis Results
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Revenue Power Analysis", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Standard Analysis", ln=True)
    pdf.set_font("Arial", "", 12)
    
    # Add Revenue results table
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(80, 10, "Parameter", 1, 0, "C", True)
    pdf.cell(80, 10, "Value", 1, 1, "C", True)
    
    for key, value in revenue_results.items():
        pdf.cell(80, 10, key, 1, 0, "L")
        if isinstance(value, float):
            pdf.cell(80, 10, f"{value:.4f}", 1, 1, "R")
        else:
            pdf.cell(80, 10, str(value), 1, 1, "R")
    
    pdf.ln(10)
    
    # Revenue Power Curve
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Revenue Power Curve", ln=True)
    
    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.figure(figsize=(10, 6))
        plot_power_curve('revenue', **revenue_results)
        plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add the image to the PDF
        pdf.image(tmp_file.name, x=10, y=None, w=190)
    
    # Remove the temporary file
    os.unlink(tmp_file.name)
    pdf.ln(10)
    
    # Custom Analyses (if provided)
    if custom_ctr_results is not None:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Custom CTR Power Analysis", ln=True)
        pdf.set_font("Arial", "", 12)
        
        # Add Custom CTR results table
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(80, 10, "Parameter", 1, 0, "C", True)
        pdf.cell(80, 10, "Value", 1, 1, "C", True)
        
        for key, value in custom_ctr_results.items():
            pdf.cell(80, 10, key, 1, 0, "L")
            if isinstance(value, float):
                pdf.cell(80, 10, f"{value:.4f}", 1, 1, "R")
            else:
                pdf.cell(80, 10, str(value), 1, 1, "R")
        
        pdf.ln(10)
        
        # Custom CTR Power Curve
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Custom CTR Power Curve", ln=True)
        
        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(10, 6))
            plot_power_curve('ctr', **custom_ctr_results)
            plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add the image to the PDF
            pdf.image(tmp_file.name, x=10, y=None, w=190)
        
        # Remove the temporary file
        os.unlink(tmp_file.name)
        pdf.ln(10)
    
    if custom_revenue_results is not None:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Custom Revenue Power Analysis", ln=True)
        pdf.set_font("Arial", "", 12)
        
        # Add Custom Revenue results table
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(80, 10, "Parameter", 1, 0, "C", True)
        pdf.cell(80, 10, "Value", 1, 1, "C", True)
        
        for key, value in custom_revenue_results.items():
            pdf.cell(80, 10, key, 1, 0, "L")
            if isinstance(value, float):
                pdf.cell(80, 10, f"{value:.4f}", 1, 1, "R")
            else:
                pdf.cell(80, 10, str(value), 1, 1, "R")
        
        pdf.ln(10)
        
        # Custom Revenue Power Curve
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Custom Revenue Power Curve", ln=True)
        
        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(10, 6))
            plot_power_curve('revenue', **custom_revenue_results)
            plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add the image to the PDF
            pdf.image(tmp_file.name, x=10, y=None, w=190)
        
        # Remove the temporary file
        os.unlink(tmp_file.name)
        pdf.ln(10)
    
    # Traffic Split Comparison (if provided)
    if traffic_split_comparison is not None:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Traffic Split Comparison", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, "This section compares the impact of different traffic splits on the required sample size and actual power.")
        
        # Add Traffic Split Comparison table
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(40, 10, "Traffic Split", 1, 0, "C", True)
        pdf.cell(40, 10, "Sample Size/Variant", 1, 0, "C", True)
        pdf.cell(40, 10, "Total Sample Size", 1, 0, "C", True)
        pdf.cell(40, 10, "Actual Power", 1, 1, "C", True)
        
        for _, row in traffic_split_comparison.iterrows():
            pdf.cell(40, 10, f"{row['traffic_split']:.2f}", 1, 0, "C")
            pdf.cell(40, 10, f"{row['sample_size_per_variant']}", 1, 0, "C")
            pdf.cell(40, 10, f"{row['total_sample_size']}", 1, 0, "C")
            pdf.cell(40, 10, f"{row['actual_power']:.4f}", 1, 1, "C")
        
        pdf.ln(10)
        
        # Traffic Split Comparison Plot
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Traffic Split Comparison Plots", ln=True)
        
        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(traffic_split_comparison['traffic_split'], traffic_split_comparison['total_sample_size'], 'o-', linewidth=2)
            plt.xlabel('Traffic Split (Treatment)')
            plt.ylabel('Total Sample Size Required')
            plt.title('Total Sample Size vs Traffic Split')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(traffic_split_comparison['traffic_split'], traffic_split_comparison['actual_power'], 'o-', linewidth=2)
            plt.xlabel('Traffic Split (Treatment)')
            plt.ylabel('Actual Power')
            plt.title('Actual Power vs Traffic Split')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add the image to the PDF
            pdf.image(tmp_file.name, x=10, y=None, w=190)
        
        # Remove the temporary file
        os.unlink(tmp_file.name)
        pdf.ln(10)
    
    # Effect Size Comparison (if provided)
    if effect_size_comparison is not None:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Effect Size Comparison", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, "This section compares the impact of different effect sizes on the required sample size and statistical power.")
        
        # Add Effect Size Comparison table
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(40, 10, "MDE ($)", 1, 0, "C", True)
        pdf.cell(40, 10, "Effect Size (d)", 1, 0, "C", True)
        pdf.cell(40, 10, "Sample Size/Variant", 1, 0, "C", True)
        pdf.cell(40, 10, "Total Sample Size", 1, 1, "C", True)
        
        for _, row in effect_size_comparison.iterrows():
            pdf.cell(40, 10, f"{row['minimum_detectable_effect']:.2f}", 1, 0, "C")
            pdf.cell(40, 10, f"{row['effect_size']:.4f}", 1, 0, "C")
            pdf.cell(40, 10, f"{row['sample_size_per_variant']}", 1, 0, "C")
            pdf.cell(40, 10, f"{row['total_sample_size']}", 1, 1, "C")
        
        pdf.ln(10)
        
        # Effect Size Comparison Plot
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Effect Size Comparison Plots", ln=True)
        
        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(effect_size_comparison['minimum_detectable_effect'], effect_size_comparison['total_sample_size'], 'o-', linewidth=2)
            plt.xlabel('Minimum Detectable Effect ($)')
            plt.ylabel('Total Sample Size Required')
            plt.title('Total Sample Size vs Effect Size')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(effect_size_comparison['minimum_detectable_effect'], effect_size_comparison['effect_size'], 'o-', linewidth=2)
            plt.xlabel('Minimum Detectable Effect ($)')
            plt.ylabel('Effect Size (Cohen\'s d)')
            plt.title('Effect Size vs Minimum Detectable Effect')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add the image to the PDF
            pdf.image(tmp_file.name, x=10, y=None, w=190)
        
        # Remove the temporary file
        os.unlink(tmp_file.name)
        pdf.ln(10)
    
    # Conclusion
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Conclusion", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This power analysis provides insights into the sample sizes required for A/B testing with different metrics, "
                "effect sizes, and traffic splits. The results can be used to plan experiments and ensure they have sufficient "
                "statistical power to detect meaningful differences between variants.")
    
    # Save the PDF
    pdf.output(output_path)
    print(f"PDF report generated: {output_path}")

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
    """Run example power analyses for both CTR and Revenue metrics and generate a PDF report."""
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
    
    # Custom analyses
    custom_ctr_results = calculate_ctr_power_analysis(
        baseline_ctr=0.05,
        minimum_detectable_effect=0.005,
        alpha=0.01,
        power=0.9,
        traffic_split=0.7
    )
    
    custom_revenue_results = calculate_revenue_power_analysis(
        baseline_mean=100.0,
        baseline_std=50.0,
        minimum_detectable_effect=10.0,
        alpha=0.01,
        power=0.9,
        traffic_split=0.7
    )
    
    # Traffic split comparison
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
    
    # Effect size comparison
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
    
    # Generate PDF report
    generate_power_analysis_pdf(
        ctr_results=ctr_results,
        revenue_results=revenue_results,
        custom_ctr_results=custom_ctr_results,
        custom_revenue_results=custom_revenue_results,
        traffic_split_comparison=split_comparison,
        effect_size_comparison=effect_comparison,
        output_path="power_analysis_report.pdf"
    )

if __name__ == "__main__":
    run_power_analysis_example()

# %%
