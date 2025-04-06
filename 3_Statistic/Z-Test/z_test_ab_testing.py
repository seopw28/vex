#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#%% Generate sample data for A/B testing
# Let's simulate an A/B test for a website conversion rate
np.random.seed(42)

# Parameters
n_a = 1000  # Number of users in group A
n_b = 1000  # Number of users in group B
p_a = 0.15  # True conversion rate for group A (15%)
p_b = 0.18  # True conversion rate for group B (18%)

# Generate binary outcomes (0 = no conversion, 1 = conversion)
group_a = np.random.binomial(1, p_a, n_a)
group_b = np.random.binomial(1, p_b, n_b)

# Create a DataFrame
ab_data = pd.DataFrame({
    'group': ['A'] * n_a + ['B'] * n_b,
    'converted': np.concatenate([group_a, group_b])
})

# Display sample data
print("Sample of A/B Test Data:")
print(ab_data.head(10))
print("\nData Summary:")
print(ab_data.groupby('group')['converted'].agg(['count', 'mean', 'sum']))

#%% Calculate conversion rates
conversion_rates = ab_data.groupby('group')['converted'].mean()
print("\nConversion Rates:")
print(conversion_rates)

#%% Visualize the data
plt.figure(figsize=(10, 6))
sns.barplot(x='group', y='converted', data=ab_data, ci=95)
plt.title('Conversion Rates by Group with 95% Confidence Intervals', fontsize=14)
plt.ylabel('Conversion Rate')
plt.xlabel('Group')
plt.ylim(0, 0.25)
plt.savefig('conversion_rates.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Perform Z-test for proportions
# Calculate the necessary statistics
n1, n2 = len(group_a), len(group_b)
p1, p2 = np.mean(group_a), np.mean(group_b)
p_combined = (np.sum(group_a) + np.sum(group_b)) / (n1 + n2)
se = np.sqrt(p_combined * (1 - p_combined) * (1/n1 + 1/n2))
z_stat = (p1 - p2) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Print results
print("\nZ-Test Results:")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

#%% Calculate effect size (Cohen's h)
# Cohen's h for proportions
h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
print(f"\nEffect size (Cohen's h): {h:.4f}")

# Interpret effect size
if abs(h) < 0.2:
    effect_size_desc = "Negligible"
elif abs(h) < 0.5:
    effect_size_desc = "Small"
elif abs(h) < 0.8:
    effect_size_desc = "Medium"
else:
    effect_size_desc = "Large"
print(f"Effect size interpretation: {effect_size_desc}")

#%% Calculate confidence intervals
# For group A
ci_a = stats.norm.interval(0.95, loc=p1, scale=np.sqrt(p1*(1-p1)/n1))
# For group B
ci_b = stats.norm.interval(0.95, loc=p2, scale=np.sqrt(p2*(1-p2)/n2))

print("\n95% Confidence Intervals:")
print(f"Group A: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
print(f"Group B: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

#%% Visualize confidence intervals
plt.figure(figsize=(10, 6))
plt.bar(['Group A', 'Group B'], [p1, p2], yerr=[p1-ci_a[0], p2-ci_b[0]], 
        capsize=10, color=['#3498db', '#e74c3c'])
plt.title('Conversion Rates with 95% Confidence Intervals', fontsize=14)
plt.ylabel('Conversion Rate')
plt.ylim(0, 0.25)
plt.savefig('confidence_intervals.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Power analysis
# Calculate the required sample size for a given power
from statsmodels.stats.power import TTestPower

# Parameters for power analysis
effect_size = abs(p2 - p1)
alpha = 0.05
power = 0.8

# Calculate required sample size
analysis = TTestPower()
required_n = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
required_n = int(np.ceil(required_n))

print(f"\nPower Analysis:")
print(f"Required sample size per group for 80% power: {required_n}")

#%% Visualize power curve
# Calculate power for different sample sizes
sample_sizes = np.arange(100, 5000, 100)
powers = [analysis.power(effect_size=effect_size, nobs=n, alpha=alpha) 
          for n in sample_sizes]

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, powers, 'b-')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% Power')
plt.axvline(x=required_n, color='g', linestyle='--', label=f'n={required_n}')
plt.title('Power Curve for Different Sample Sizes', fontsize=14)
plt.xlabel('Sample Size per Group')
plt.ylabel('Power')
plt.grid(True)
plt.legend()
plt.savefig('power_curve.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Calculate minimum detectable effect (MDE)
# For a given sample size and power
def calculate_mde(n, power=0.8, alpha=0.05):
    analysis = TTestPower()
    mde = analysis.solve_power(power=power, nobs=n, alpha=alpha)
    return mde

# Calculate MDE for different sample sizes
mde_values = [calculate_mde(n) for n in sample_sizes]

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, mde_values, 'g-')
plt.title('Minimum Detectable Effect for Different Sample Sizes', fontsize=14)
plt.xlabel('Sample Size per Group')
plt.ylabel('Minimum Detectable Effect')
plt.grid(True)
plt.savefig('mde_curve.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Business impact analysis
# Calculate potential revenue impact
average_order_value = 50  # Average order value in dollars
current_conversion_rate = p1
new_conversion_rate = p2
monthly_visitors = 10000  # Estimated monthly visitors

# Calculate monthly revenue
current_revenue = monthly_visitors * current_conversion_rate * average_order_value
new_revenue = monthly_visitors * new_conversion_rate * average_order_value
revenue_increase = new_revenue - current_revenue
revenue_increase_percentage = (revenue_increase / current_revenue) * 100

print("\nBusiness Impact Analysis:")
print(f"Current monthly revenue: ${current_revenue:,.2f}")
print(f"Projected monthly revenue: ${new_revenue:,.2f}")
print(f"Monthly revenue increase: ${revenue_increase:,.2f} ({revenue_increase_percentage:.2f}%)")

#%% Visualize business impact
plt.figure(figsize=(10, 6))
plt.bar(['Current', 'Projected'], [current_revenue, new_revenue], 
        color=['#3498db', '#2ecc71'])
plt.title('Monthly Revenue Impact', fontsize=14)
plt.ylabel('Revenue ($)')
plt.text(0, current_revenue, f'${current_revenue:,.0f}', ha='center', va='bottom')
plt.text(1, new_revenue, f'${new_revenue:,.0f}', ha='center', va='bottom')
plt.text(0.5, (current_revenue + new_revenue)/2, 
         f'+${revenue_increase:,.0f}\n({revenue_increase_percentage:.1f}%)', 
         ha='center', va='center', fontsize=12)
plt.savefig('revenue_impact.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Conclusion and recommendations
print("\nConclusion and Recommendations:")
if p_value < 0.05:
    print("There is a statistically significant difference between the two groups.")
    if p2 > p1:
        print(f"Group B outperforms Group A with a {((p2/p1)-1)*100:.1f}% increase in conversion rate.")
    else:
        print(f"Group A outperforms Group B with a {((p1/p2)-1)*100:.1f}% increase in conversion rate.")
else:
    print("There is no statistically significant difference between the two groups.")
    print("Consider running the test longer or with a larger sample size.")

# Ensure effect_size_desc is defined
if 'effect_size_desc' not in locals():
    # Recalculate Cohen's h if needed
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    # Interpret effect size
    if abs(h) < 0.2:
        effect_size_desc = "Negligible"
    elif abs(h) < 0.5:
        effect_size_desc = "Small"
    elif abs(h) < 0.8:
        effect_size_desc = "Medium"
    else:
        effect_size_desc = "Large"

print(f"\nThe effect size is {effect_size_desc}, indicating a {effect_size_desc.lower()} practical significance.")
print(f"The business impact is estimated at ${revenue_increase:,.2f} per month.") 
# %%
