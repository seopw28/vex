#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import TTestPower
import warnings
warnings.filterwarnings('ignore')

#%% Generate sample A/B test data
# Let's simulate an A/B test for a website conversion rate
np.random.seed(42)

# Parameters
n_a = 1000  # Number of users in group A
n_b = 1000  # Number of users in group B
mean_a = 0.15  # True mean conversion rate for group A (15%)
mean_b = 0.18  # True mean conversion rate for group B (18%)
std_a = 0.05   # Standard deviation for group A
std_b = 0.05   # Standard deviation for group B

# Generate data
conversion_a = np.random.normal(mean_a, std_a, n_a)
conversion_b = np.random.normal(mean_b, std_b, n_b)

# Ensure values are between 0 and 1 (conversion rates)
conversion_a = np.clip(conversion_a, 0, 1)
conversion_b = np.clip(conversion_b, 0, 1)

# Create DataFrame
ab_data = pd.DataFrame({
    'Group': ['A'] * n_a + ['B'] * n_b,
    'Conversion_Rate': np.concatenate([conversion_a, conversion_b])
})

# Display sample data
print("Sample of A/B Test Data:")
print(ab_data.head(10))
print("\nData Summary:")
print(ab_data.groupby('Group')['Conversion_Rate'].describe())

#%% Visualize A/B test data
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Conversion_Rate', data=ab_data)
sns.swarmplot(x='Group', y='Conversion_Rate', data=ab_data, color='red', alpha=0.3)
plt.title('Conversion Rates by Group', fontsize=14)
plt.xlabel('Group')
plt.ylabel('Conversion Rate')
plt.savefig('ab_test_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Calculate Cohen's d effect size
def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size for two independent groups.
    
    Parameters:
    -----------
    group1 : array-like
        Data for the first group
    group2 : array-like
        Data for the second group
        
    Returns:
    --------
    d : float
        Cohen's d effect size
    pooled_std : float
        Pooled standard deviation
    """
    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    # Calculate standard deviations
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    
    # Calculate sample sizes
    n1 = len(group1)
    n2 = len(group2)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d, pooled_std

# Calculate Cohen's d for our A/B test data
group_a = ab_data[ab_data['Group'] == 'A']['Conversion_Rate']
group_b = ab_data[ab_data['Group'] == 'B']['Conversion_Rate']
d, pooled_std = cohens_d(group_a, group_b)

print("\nCohen's d Effect Size:")
print(f"Cohen's d: {d:.4f}")

# Interpret Cohen's d
def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size.
    
    Parameters:
    -----------
    d : float
        Cohen's d effect size
        
    Returns:
    --------
    interpretation : str
        Text interpretation of the effect size
    """
    if abs(d) < 0.2:
        return "negligible"
    elif abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"

effect_size_desc = interpret_cohens_d(d)
print(f"Effect size interpretation: {effect_size_desc}")

#%% Perform t-test to check for statistical significance
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True)
print("\nIndependent t-test results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")

#%% Calculate confidence interval for Cohen's d
def cohens_d_ci(d, n1, n2, conf_level=0.95):
    """
    Calculate confidence interval for Cohen's d.
    
    Parameters:
    -----------
    d : float
        Cohen's d effect size
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2
    conf_level : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    """
    # Calculate standard error of d
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2)))
    
    # Calculate critical value for the confidence level
    z_crit = stats.norm.ppf((1 + conf_level) / 2)
    
    # Calculate confidence interval
    ci_lower = d - z_crit * se
    ci_upper = d + z_crit * se
    
    return ci_lower, ci_upper

# Calculate confidence interval
ci_lower, ci_upper = cohens_d_ci(d, n_a, n_b)
print(f"\n95% Confidence Interval for Cohen's d: [{ci_lower:.4f}, {ci_upper:.4f}]")

#%% Visualize Cohen's d with confidence interval
plt.figure(figsize=(10, 6))
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=d, color='blue', linewidth=2, label=f"Cohen's d = {d:.4f}")
plt.axvspan(ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')

# Add interpretation markers
interpretations = {
    0.2: 'Negligible',
    0.5: 'Small',
    0.8: 'Medium',
    1.3: 'Large'
}

for threshold, label in interpretations.items():
    plt.axvline(x=threshold, color='red', linestyle=':', alpha=0.5)
    plt.axvline(x=-threshold, color='red', linestyle=':', alpha=0.5)
    plt.text(threshold, 0.1, label, rotation=90, va='bottom', ha='right')
    plt.text(-threshold, 0.1, label, rotation=90, va='bottom', ha='left')

plt.title("Cohen's d Effect Size with Confidence Interval", fontsize=14)
plt.xlabel("Cohen's d")
plt.ylabel("Density")
plt.xlim(-1.5, 1.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cohens_d_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Power analysis for Cohen's d
def power_analysis_cohens_d(d, alpha=0.05, power=0.8):
    """
    Perform power analysis for Cohen's d.
    
    Parameters:
    -----------
    d : float
        Cohen's d effect size
    alpha : float
        Significance level (default: 0.05)
    power : float
        Desired power (default: 0.8)
        
    Returns:
    --------
    n_per_group : int
        Required sample size per group
    """
    # Use statsmodels power analysis
    analysis = TTestPower()
    
    # Calculate required sample size
    n_per_group = analysis.solve_power(effect_size=d, 
                                      alpha=alpha, 
                                      power=power)
    
    # Round up to the nearest integer
    n_per_group = int(np.ceil(n_per_group))
    
    return n_per_group

# Calculate required sample size for 80% power
required_n = power_analysis_cohens_d(d)
print(f"\nPower Analysis:")
print(f"Required sample size per group for 80% power: {required_n}")

# Calculate power for a range of sample sizes
n_range = np.arange(10, 2000, 100)
power_values = []

for n in n_range:
    power_value = TTestPower().power(effect_size=d, 
                                     nobs=n, 
                                     alpha=0.05)
    power_values.append(power_value)

# Visualize power curve
plt.figure(figsize=(10, 6))
plt.plot(n_range, power_values, 'b-', linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')
plt.axvline(x=required_n, color='g', linestyle='--', label=f'Required n={required_n}')
plt.title('Power Curve for Cohen\'s d', fontsize=14)
plt.xlabel('Sample Size per Group')
plt.ylabel('Power')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('power_curve.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Conclusion
print("\nConclusion:")
print(f"1. The A/B test shows a {effect_size_desc} effect size (Cohen's d = {d:.4f})")
print(f"   - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"2. The difference between groups is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} (p = {p_value:.4f})")
print(f"3. To achieve 80% power, {required_n} participants per group would be needed")
print(f"4. Based on the current sample size (n={n_a} per group), the power is {power_values[0]:.4f}")

#%% Additional analysis: Non-parametric comparison
# Mann-Whitney U test (non-parametric alternative to t-test)
u_stat, p_value_mw = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
print("\nMann-Whitney U test (non-parametric):")
print(f"U-statistic: {u_stat:.4f}")
print(f"p-value: {p_value_mw:.4f}")
print(f"Statistically significant: {'Yes' if p_value_mw < 0.05 else 'No'}")

#%% Additional analysis: Bootstrap confidence interval for Cohen's d
def bootstrap_cohens_d_ci(group1, group2, n_bootstrap=10000, conf_level=0.95):
    """
    Calculate bootstrap confidence interval for Cohen's d.
    
    Parameters:
    -----------
    group1 : array-like
        Data for the first group
    group2 : array-like
        Data for the second group
    n_bootstrap : int
        Number of bootstrap samples
    conf_level : float
        Confidence level
        
    Returns:
    --------
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    bootstrap_ds : array
        Array of bootstrap Cohen's d values
    """
    bootstrap_ds = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        
        # Calculate Cohen's d for this bootstrap sample
        d, _ = cohens_d(sample1, sample2)
        bootstrap_ds[i] = d
    
    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_ds, ((1 - conf_level) / 2) * 100)
    ci_upper = np.percentile(bootstrap_ds, (1 - ((1 - conf_level) / 2)) * 100)
    
    return ci_lower, ci_upper, bootstrap_ds

# Calculate bootstrap confidence interval
bootstrap_ci_lower, bootstrap_ci_upper, bootstrap_ds = bootstrap_cohens_d_ci(group_a, group_b)

print("\nBootstrap Confidence Interval for Cohen's d:")
print(f"95% CI: [{bootstrap_ci_lower:.4f}, {bootstrap_ci_upper:.4f}]")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_ds, kde=True)
plt.axvline(x=d, color='red', linestyle='--', label=f"Observed d = {d:.4f}")
plt.axvline(x=bootstrap_ci_lower, color='blue', linestyle=':', label='95% CI Lower')
plt.axvline(x=bootstrap_ci_upper, color='blue', linestyle=':', label='95% CI Upper')
plt.title("Bootstrap Distribution of Cohen's d", fontsize=14)
plt.xlabel("Cohen's d")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bootstrap_cohens_d.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Additional analysis: Effect size for proportions
# For binary outcomes (e.g., converted vs. not converted)
def cohens_h(p1, p2, n1, n2):
    """
    Calculate Cohen's h effect size for proportions.
    
    Parameters:
    -----------
    p1 : float
        Proportion for group 1
    p2 : float
        Proportion for group 2
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2
        
    Returns:
    --------
    h : float
        Cohen's h effect size
    """
    # Calculate arcsine transformation
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    
    # Calculate Cohen's h
    h = phi1 - phi2
    
    return h

# Calculate proportions
p_a = np.mean(group_a)
p_b = np.mean(group_b)

# Calculate Cohen's h
h = cohens_h(p_a, p_b, n_a, n_b)
print(f"\nCohen's h for proportions: {h:.4f}")

# Interpret Cohen's h
def interpret_cohens_h(h):
    """
    Interpret Cohen's h effect size.
    
    Parameters:
    -----------
    h : float
        Cohen's h effect size
        
    Returns:
    --------
    interpretation : str
        Text interpretation of the effect size
    """
    if abs(h) < 0.2:
        return "negligible"
    elif abs(h) < 0.5:
        return "small"
    elif abs(h) < 0.8:
        return "medium"
    else:
        return "large"

effect_size_desc_h = interpret_cohens_h(h)
print(f"Effect size interpretation (Cohen's h): {effect_size_desc_h}")

#%% Final summary
print("\nFinal Summary:")
print(f"1. Continuous measure (Cohen's d): {d:.4f} ({effect_size_desc} effect)")
print(f"   - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"   - Bootstrap 95% CI: [{bootstrap_ci_lower:.4f}, {bootstrap_ci_upper:.4f}]")
print(f"2. Binary outcome (Cohen's h): {h:.4f} ({effect_size_desc_h} effect)")
print(f"3. Statistical significance: {'Yes' if p_value < 0.05 else 'No'} (p = {p_value:.4f})")
print(f"4. Required sample size for 80% power: {required_n} per group")
print(f"5. Current power: {power_values[0]:.4f}") 
# %%
