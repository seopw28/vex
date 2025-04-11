#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
import pandas as pd
from matplotlib.patches import Patch

#%%
# Function to calculate Cohen's h
def cohens_h(p1, p2):
    """
    Calculate Cohen's h effect size for two proportions.
    
    Parameters:
    -----------
    p1 : float
        First proportion (between 0 and 1)
    p2 : float
        Second proportion (between 0 and 1)
        
    Returns:
    --------
    h : float
        Cohen's h effect size
    """
    # Ensure proportions are valid
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("Proportions must be between 0 and 1")
    
    # Calculate Cohen's h
    h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
    
    return h

#%%
# Function to interpret Cohen's h
def interpret_cohens_h(h):
    """
    Interpret the magnitude of Cohen's h effect size.
    
    Parameters:
    -----------
    h : float
        Cohen's h effect size
        
    Returns:
    --------
    interpretation : str
        Interpretation of the effect size
    """
    h_abs = abs(h)
    
    if h_abs < 0.2:
        return "Negligible effect"
    elif h_abs < 0.5:
        return "Small effect"
    elif h_abs < 0.8:
        return "Medium effect"
    else:
        return "Large effect"

#%%
# Sample data: Success rates for two treatments
# Treatment A
successes_A = 120
total_A = 200
p_A = successes_A / total_A

# Treatment B
successes_B = 150
total_B = 200
p_B = successes_B / total_B

# Calculate Cohen's h
h = cohens_h(p_A, p_B)
interpretation = interpret_cohens_h(h)

# Print results
print(f"Treatment A success rate: {p_A:.4f} ({successes_A}/{total_A})")
print(f"Treatment B success rate: {p_B:.4f} ({successes_B}/{total_B})")
print(f"Cohen's h: {h:.4f}")
print(f"Interpretation: {interpretation}")

#%%
# Calculate confidence interval for Cohen's h using bootstrap
def bootstrap_cohens_h_ci(successes1, total1, successes2, total2, n_bootstrap=10000, ci=0.95):
    """
    Calculate confidence interval for Cohen's h using bootstrap.
    
    Parameters:
    -----------
    successes1, successes2 : int
        Number of successes in each group
    total1, total2 : int
        Total number of trials in each group
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence level (between 0 and 1)
        
    Returns:
    --------
    lower, upper : float
        Lower and upper bounds of the confidence interval
    """
    p1 = successes1 / total1
    p2 = successes2 / total2
    
    # Generate bootstrap samples
    bootstrap_h = []
    
    for _ in range(n_bootstrap):
        # Resample from binomial distributions
        sample1 = np.random.binomial(total1, p1) / total1
        sample2 = np.random.binomial(total2, p2) / total2
        
        # Calculate Cohen's h for this bootstrap sample
        h_sample = cohens_h(sample1, sample2)
        bootstrap_h.append(h_sample)
    
    # Calculate confidence interval
    alpha = 1 - ci
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(bootstrap_h, lower_percentile)
    upper = np.percentile(bootstrap_h, upper_percentile)
    
    return lower, upper

#%%
# Calculate confidence interval
lower_ci, upper_ci = bootstrap_cohens_h_ci(successes_A, total_A, successes_B, total_B)
print(f"95% Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")

#%%
# Visualization 1: Bar chart of proportions
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
treatments = ['Treatment A', 'Treatment B']
proportions = [p_A, p_B]
colors = ['#3498db', '#e74c3c']

plt.bar(treatments, proportions, color=colors)
plt.ylim(0, 1)
plt.ylabel('Success Rate')
plt.title('Comparison of Success Rates')

for i, prop in enumerate(proportions):
    plt.text(i, prop + 0.02, f'{prop:.2f}', ha='center', va='bottom')

#%%
# Visualization 2: Cohen's h with confidence interval
plt.subplot(2, 2, 2)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.3)

plt.errorbar(['Cohen\'s h'], [h], yerr=[[h-lower_ci], [upper_ci-h]], 
             fmt='o', color='#9b59b6', ecolor='#9b59b6', capsize=10, markersize=10)
plt.ylim(-1, 1)
plt.title('Cohen\'s h Effect Size with 95% CI')

# Add text annotations for interpretation regions
plt.text(1.1, 0.1, 'Negligible effect', ha='left', va='center')
plt.text(1.1, 0.35, 'Small effect', ha='left', va='center')
plt.text(1.1, 0.65, 'Medium effect', ha='left', va='center')
plt.text(1.1, 0.9, 'Large effect', ha='left', va='center')
plt.text(1.1, -0.1, 'Negligible effect', ha='left', va='center')
plt.text(1.1, -0.35, 'Small effect', ha='left', va='center')
plt.text(1.1, -0.65, 'Medium effect', ha='left', va='center')
plt.text(1.1, -0.9, 'Large effect', ha='left', va='center')

#%%
# Visualization 3: Bootstrap distribution
plt.subplot(2, 2, 3)
bootstrap_samples = [cohens_h(np.random.binomial(total_A, p_A) / total_A, 
                             np.random.binomial(total_B, p_B) / total_B) 
                    for _ in range(5000)]

sns.histplot(bootstrap_samples, kde=True, color='#2ecc71')
plt.axvline(x=h, color='red', linestyle='-', label='Observed h')
plt.axvline(x=lower_ci, color='black', linestyle='--', label='95% CI')
plt.axvline(x=upper_ci, color='black', linestyle='--')
plt.xlabel('Cohen\'s h')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Cohen\'s h')
plt.legend()

#%%
# Visualization 4: Effect size interpretation
plt.subplot(2, 2, 4)

# Create a visual representation of effect size
effect_sizes = np.linspace(-1, 1, 1000)
categories = [interpret_cohens_h(es) for es in effect_sizes]
category_colors = {
    "Negligible effect": "#f1c40f",
    "Small effect": "#3498db",
    "Medium effect": "#2ecc71",
    "Large effect": "#e74c3c"
}
colors = [category_colors[cat] for cat in categories]

# Create a colorbar-like visualization
plt.scatter(effect_sizes, [1]*len(effect_sizes), c=colors, marker='|', s=100)
plt.axvline(x=h, color='black', linestyle='-', linewidth=2, label=f'Observed h = {h:.2f}')
plt.axvline(x=lower_ci, color='gray', linestyle='--', label=f'95% CI: [{lower_ci:.2f}, {upper_ci:.2f}]')
plt.axvline(x=upper_ci, color='gray', linestyle='--')

# Add labels for effect size regions
plt.text(-0.9, 1.1, "Large\nnegative", ha='center', va='bottom')
plt.text(-0.65, 1.1, "Medium\nnegative", ha='center', va='bottom')
plt.text(-0.35, 1.1, "Small\nnegative", ha='center', va='bottom')
plt.text(0, 1.1, "Negligible", ha='center', va='bottom')
plt.text(0.35, 1.1, "Small\npositive", ha='center', va='bottom')
plt.text(0.65, 1.1, "Medium\npositive", ha='center', va='bottom')
plt.text(0.9, 1.1, "Large\npositive", ha='center', va='bottom')

# Create legend for effect size categories
legend_elements = [
    Patch(facecolor=category_colors["Negligible effect"], label="Negligible effect"),
    Patch(facecolor=category_colors["Small effect"], label="Small effect"),
    Patch(facecolor=category_colors["Medium effect"], label="Medium effect"),
    Patch(facecolor=category_colors["Large effect"], label="Large effect")
]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

plt.xlim(-1, 1)
plt.ylim(0.9, 1.2)
plt.title('Cohen\'s h Effect Size Interpretation')
plt.yticks([])
plt.xlabel('Cohen\'s h')

plt.tight_layout()
plt.savefig('cohens_h_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'cohens_h_visualization.png'")

#%%
# Additional analysis: Power analysis for Cohen's h
def power_analysis_cohens_h(h, alpha=0.05, power=0.8):
    """
    Calculate required sample size for a given Cohen's h, alpha, and power.
    
    Parameters:
    -----------
    h : float
        Cohen's h effect size
    alpha : float
        Significance level
    power : float
        Desired statistical power
        
    Returns:
    --------
    n : int
        Required sample size per group
    """
    # Calculate z-scores for alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_power = stats.norm.ppf(power)
    
    # Calculate required sample size
    n = 2 * ((z_alpha + z_power) / h) ** 2
    
    return math.ceil(n)

#%%
# Calculate required sample size
required_n = power_analysis_cohens_h(h)
print(f"\nPower Analysis:")
print(f"To detect an effect size of h = {h:.4f} with 80% power at alpha = 0.05,")
print(f"you would need approximately {required_n} participants per group.")

#%%
# Create power curve
effect_sizes = np.linspace(0.1, 1.0, 10)
sample_sizes = [power_analysis_cohens_h(es) for es in effect_sizes]

plt.figure(figsize=(10, 6))
plt.plot(effect_sizes, sample_sizes, marker='o', linestyle='-', color='#3498db')
plt.axvline(x=h, color='red', linestyle='--', label=f'Current h = {h:.2f}')
plt.axhline(y=total_A, color='green', linestyle='--', label=f'Current n = {total_A}')
plt.xlabel('Cohen\'s h Effect Size')
plt.ylabel('Required Sample Size per Group')
plt.title('Sample Size Requirements for Different Effect Sizes')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('cohens_h_power_analysis.png', dpi=300)
plt.show()

print("\nPower analysis visualization saved as 'cohens_h_power_analysis.png'")
#%%

