#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.power import TTestPower

# Set random seed for reproducibility
np.random.seed(42)

#%%
# Function to generate sample data for A/B testing
def generate_ab_test_data(n_samples=1000, effect_size=0.2, noise_level=0.5):
    """
    Generate sample data for A/B testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples per group (A and B)
    effect_size : float
        The true effect size between groups (standardized difference)
    noise_level : float
        Level of noise in the data
        
    Returns:
    --------
    data : pandas.DataFrame
        DataFrame containing the A/B test data
    """
    # Generate data for group A (control)
    group_a = np.random.normal(0, 1, n_samples)
    
    # Generate data for group B (treatment) with effect size
    group_b = np.random.normal(effect_size, 1, n_samples)
    
    # Add some noise
    group_a += np.random.normal(0, noise_level, n_samples)
    group_b += np.random.normal(0, noise_level, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'group': ['A'] * n_samples + ['B'] * n_samples,
        'value': np.concatenate([group_a, group_b])
    })
    
    return data

#%%
# Generate sample data
print("Generating sample A/B test data...")
ab_data = generate_ab_test_data(n_samples=1000, effect_size=0.2, noise_level=0.5)

# Display the first few rows
print("\nFirst few rows of the data:")
display(ab_data.head())

#%%
# Basic descriptive statistics
print("\nDescriptive statistics by group:")
display(ab_data.groupby('group').describe())

#%%
# Visualize the data
plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='value', data=ab_data)
plt.title('A/B Test Data Distribution')
plt.xlabel('Group')
plt.ylabel('Value')
plt.savefig('ab_test_boxplot.png')
print("Box plot saved to 'ab_test_boxplot.png'")

#%%
# Perform t-test
print("\nPerforming t-test...")
group_a_values = ab_data[ab_data['group'] == 'A']['value']
group_b_values = ab_data[ab_data['group'] == 'B']['value']

# Independent samples t-test
t_stat, p_value = stats.ttest_ind(group_a_values, group_b_values)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print(f"Result: Statistically significant difference (p < {alpha})")
    print("We reject the null hypothesis and conclude there is a significant difference between groups.")
else:
    print(f"Result: No statistically significant difference (p >= {alpha})")
    print("We fail to reject the null hypothesis and cannot conclude there is a significant difference between groups.")

#%%
# Calculate effect size (Cohen's d)
def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
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
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_se
    
    return d

# Calculate effect size
effect_size = cohens_d(group_a_values, group_b_values)
print(f"\nEffect size (Cohen's d): {effect_size:.4f}")

# Interpret effect size
if abs(effect_size) < 0.2:
    print("Effect size interpretation: Negligible")
elif abs(effect_size) < 0.5:
    print("Effect size interpretation: Small")
elif abs(effect_size) < 0.8:
    print("Effect size interpretation: Medium")
else:
    print("Effect size interpretation: Large")

#%%
# Power analysis
print("\nPerforming power analysis...")
# Parameters for power analysis
effect_size_abs = abs(effect_size)
alpha = 0.05
nobs = len(group_a_values)

# Calculate power
power_analysis = TTestPower()
power = power_analysis.power(effect_size=effect_size_abs, nobs=nobs, alpha=alpha)

print(f"Statistical power: {power:.4f}")

# Interpret power
if power < 0.5:
    print("Power interpretation: Low power, high risk of Type II error")
elif power < 0.8:
    print("Power interpretation: Moderate power")
else:
    print("Power interpretation: High power")

#%%
# Calculate required sample size for desired power
desired_power = 0.8
required_nobs = power_analysis.solve_power(effect_size=effect_size_abs, power=desired_power, alpha=alpha)

print(f"\nRequired sample size per group for {desired_power*100}% power: {required_nobs:.0f}")

#%%
# Create a visualization of the t-test
plt.figure(figsize=(12, 6))

# Plot 1: Data distribution
plt.subplot(1, 2, 1)
sns.histplot(data=ab_data, x='value', hue='group', kde=True, bins=30)
plt.title('Distribution of Values by Group')
plt.xlabel('Value')
plt.ylabel('Count')

# Plot 2: T-test visualization
plt.subplot(1, 2, 2)
# Calculate mean and standard error for each group
group_means = ab_data.groupby('group')['value'].mean()
group_sems = ab_data.groupby('group')['value'].sem()

# Create error bar plot
plt.errorbar(
    x=['A', 'B'],
    y=group_means,
    yerr=group_sems,
    fmt='o',
    capsize=5,
    capthick=2,
    markersize=8,
    color='blue'
)

# Add significance indicator if p-value is significant
if p_value < 0.05:
    # Calculate y position for the significance line
    y_max = max(group_means) + max(group_sems) + 0.5
    
    # Draw significance line
    plt.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
    plt.text(0.5, y_max + 0.1, '*', ha='center', fontsize=14)

plt.title('T-Test Results')
plt.xlabel('Group')
plt.ylabel('Mean Value')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('ab_test_results.png')
print("T-test visualization saved to 'ab_test_results.png'")

#%%
# Create a detailed report
print("\n=== A/B TEST REPORT ===")
print(f"Sample size per group: {nobs}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Effect size (Cohen's d): {effect_size:.4f}")
print(f"Statistical power: {power:.4f}")
print(f"Required sample size for 80% power: {required_nobs:.0f}")

# Save the data to CSV
ab_data.to_csv('ab_test_data.csv', index=False)
print("\nData saved to 'ab_test_data.csv'")

#%%
# Function to perform A/B test with different parameters
def run_ab_test(n_samples=1000, effect_size=0.2, noise_level=0.5, alpha=0.05):
    """
    Run a complete A/B test analysis.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples per group
    effect_size : float
        The true effect size between groups
    noise_level : float
        Level of noise in the data
    alpha : float
        Significance level
        
    Returns:
    --------
    results : dict
        Dictionary containing test results
    """
    # Generate data
    data = generate_ab_test_data(n_samples, effect_size, noise_level)
    
    # Split data by group
    group_a = data[data['group'] == 'A']['value']
    group_b = data[data['group'] == 'B']['value']
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    
    # Calculate effect size
    d = cohens_d(group_a, group_b)
    
    # Calculate power
    power = TTestPower().power(effect_size=abs(d), nobs=n_samples, alpha=alpha)
    
    # Compile results
    results = {
        'n_samples': n_samples,
        'effect_size': effect_size,
        'noise_level': noise_level,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': d,
        'power': power,
        'significant': p_value < alpha
    }
    
    return results, data

#%%
# Run multiple A/B tests with different parameters
print("Running multiple A/B tests with different parameters...")

# Test scenarios
scenarios = [
    {'n_samples': 100, 'effect_size': 0.2, 'noise_level': 0.5},
    {'n_samples': 500, 'effect_size': 0.2, 'noise_level': 0.5},
    {'n_samples': 1000, 'effect_size': 0.2, 'noise_level': 0.5},
    {'n_samples': 1000, 'effect_size': 0.1, 'noise_level': 0.5},
    {'n_samples': 1000, 'effect_size': 0.5, 'noise_level': 0.5},
    {'n_samples': 1000, 'effect_size': 0.2, 'noise_level': 0.2},
    {'n_samples': 1000, 'effect_size': 0.2, 'noise_level': 1.0}
]

# Run tests and collect results
all_results = []
for scenario in scenarios:
    results, _ = run_ab_test(**scenario)
    all_results.append(results)

# Convert to DataFrame
results_df = pd.DataFrame(all_results)
display(results_df)

#%%
# Visualize the impact of sample size and effect size on power
print("Visualizing the impact of sample size and effect size on power...")

# Create a grid of sample sizes and effect sizes
sample_sizes = np.array([50, 100, 200, 500, 1000, 2000])
effect_sizes = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])

# Calculate power for each combination
power_matrix = np.zeros((len(sample_sizes), len(effect_sizes)))
for i, n in enumerate(sample_sizes):
    for j, d in enumerate(effect_sizes):
        power_matrix[i, j] = TTestPower().power(effect_size=d, nobs=n, alpha=0.05)

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    power_matrix,
    annot=True,
    fmt='.2f',
    cmap='YlGnBu',
    xticklabels=effect_sizes,
    yticklabels=sample_sizes,
    cbar_kws={'label': 'Power'}
)
plt.title('Statistical Power by Sample Size and Effect Size')
plt.xlabel('Effect Size (Cohen\'s d)')
plt.ylabel('Sample Size per Group')
plt.tight_layout()
plt.savefig('power_analysis_heatmap.png')
print("Power analysis heatmap saved to 'power_analysis_heatmap.png'")

#%%
# Main function to run the analysis
def main():
    # Generate sample data
    print("Generating sample A/B test data...")
    ab_data = generate_ab_test_data(n_samples=1000, effect_size=0.2, noise_level=0.5)
    
    # Perform t-test
    group_a_values = ab_data[ab_data['group'] == 'A']['value']
    group_b_values = ab_data[ab_data['group'] == 'B']['value']
    t_stat, p_value = stats.ttest_ind(group_a_values, group_b_values)
    
    # Calculate effect size
    effect_size = cohens_d(group_a_values, group_b_values)
    
    # Calculate power
    power = TTestPower().power(effect_size=abs(effect_size), nobs=len(group_a_values), alpha=0.05)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='value', data=ab_data)
    plt.title('A/B Test Data Distribution')
    plt.xlabel('Group')
    plt.ylabel('Value')
    plt.savefig('ab_test_boxplot.png')
    
    # Save the data to CSV
    ab_data.to_csv('ab_test_data.csv', index=False)
    
    print("A/B test analysis completed successfully!")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    print(f"Statistical power: {power:.4f}")

if __name__ == "__main__":
    main() 