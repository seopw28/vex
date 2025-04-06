#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pymc as pm
import arviz as az
from fpdf import FPDF
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#%% Generate sample data for A/B testing
def generate_ab_test_data(n_samples_a=200, n_samples_b=200, conversion_rate_a=0.15, 
                         conversion_rate_b=0.18, random_state=42):
    """
    Generate synthetic A/B test data.
    
    Parameters:
    -----------
    n_samples_a : int
        Number of samples for variant A
    n_samples_b : int
        Number of samples for variant B
    conversion_rate_a : float
        True conversion rate for variant A
    conversion_rate_b : float
        True conversion rate for variant B
    random_state : int
        Random seed
        
    Returns:
    --------
    data : pandas.DataFrame
        DataFrame containing the A/B test data
    """
    np.random.seed(random_state)
    
    # Generate data for variant A
    conversions_a = np.random.binomial(1, conversion_rate_a, n_samples_a)
    
    # Generate data for variant B
    conversions_b = np.random.binomial(1, conversion_rate_b, n_samples_b)
    
    # Create DataFrame
    data = pd.DataFrame({
        'variant': ['A'] * n_samples_a + ['B'] * n_samples_b,
        'converted': np.concatenate([conversions_a, conversions_b])
    })
    
    return data

# Generate sample data
data = generate_ab_test_data()

# Display sample data
print("Sample of A/B Test Data:")
print(data.head(10))
print("\nSummary Statistics:")
print(data.groupby('variant')['converted'].agg(['count', 'mean', 'sum']))

#%% Visualize the data
plt.figure(figsize=(8, 5))
sns.countplot(x='variant', hue='converted', data=data, palette='viridis')
plt.title('Conversion Distribution by Variant', fontsize=14)
plt.xlabel('Variant')
plt.ylabel('Count')
plt.legend(title='Converted')
plt.savefig('ab_test_conversion_distribution.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Frequentist A/B test (for comparison)
# Calculate conversion rates
conversion_rates = data.groupby('variant')['converted'].mean()
print("\nConversion Rates:")
print(conversion_rates)

# Perform chi-square test
contingency_table = pd.crosstab(data['variant'], data['converted'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-square Test Results:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

#%% Bayesian A/B test - Ultra Simplified
# Prepare data
n_a = data[data['variant'] == 'A']['converted'].sum()
n_b = data[data['variant'] == 'B']['converted'].sum()
total_a = len(data[data['variant'] == 'A'])
total_b = len(data[data['variant'] == 'B'])

# Define the model - Using a simpler model with fewer parameters
with pm.Model() as ab_model:
    # Priors for conversion rates
    p_a = pm.Beta('p_a', alpha=1, beta=1)
    p_b = pm.Beta('p_b', alpha=1, beta=1)
    
    # Likelihood - Using summary statistics instead of raw data
    obs_a = pm.Binomial('obs_a', n=total_a, p=p_a, observed=n_a)
    obs_b = pm.Binomial('obs_b', n=total_b, p=p_b, observed=n_b)
    
    # Difference between conversion rates
    diff = pm.Deterministic('diff', p_b - p_a)
    
    # Relative improvement
    rel_improvement = pm.Deterministic('rel_improvement', (p_b - p_a) / p_a)

# Sample from the posterior - Minimal sampling
with ab_model:
    # Use minimal samples and chains for faster execution
    trace = pm.sample(200, tune=100, chains=1, return_inferencedata=True, random_seed=42)

#%% Analyze the results
# Extract posterior samples
p_a_samples = trace.posterior['p_a'].values.flatten()
p_b_samples = trace.posterior['p_b'].values.flatten()
diff_samples = trace.posterior['diff'].values.flatten()
rel_improvement_samples = trace.posterior['rel_improvement'].values.flatten()

# Calculate summary statistics
print("\nBayesian A/B Test Results:")
print(f"Variant A conversion rate: {np.mean(p_a_samples):.4f} (95% CI: {np.percentile(p_a_samples, 2.5):.4f} - {np.percentile(p_a_samples, 97.5):.4f})")
print(f"Variant B conversion rate: {np.mean(p_b_samples):.4f} (95% CI: {np.percentile(p_b_samples, 2.5):.4f} - {np.percentile(p_b_samples, 97.5):.4f})")
print(f"Absolute difference (B - A): {np.mean(diff_samples):.4f} (95% CI: {np.percentile(diff_samples, 2.5):.4f} - {np.percentile(diff_samples, 97.5):.4f})")
print(f"Relative improvement: {np.mean(rel_improvement_samples):.2%} (95% CI: {np.percentile(rel_improvement_samples, 2.5):.2%} - {np.percentile(rel_improvement_samples, 97.5):.2%})")

# Calculate probability that B is better than A
prob_b_better = np.mean(diff_samples > 0)
print(f"Probability that B is better than A: {prob_b_better:.2%}")

#%% Visualize the posterior distributions - Ultra Simplified
# Plot posterior distributions - Only the most important ones
plt.figure(figsize=(10, 6))

# Conversion rates
plt.subplot(1, 2, 1)
sns.histplot(p_a_samples, kde=True, color='blue', alpha=0.5, label='Variant A')
sns.histplot(p_b_samples, kde=True, color='green', alpha=0.5, label='Variant B')
plt.axvline(np.mean(p_a_samples), color='blue', linestyle='--', alpha=0.8)
plt.axvline(np.mean(p_b_samples), color='green', linestyle='--', alpha=0.8)
plt.title('Posterior Distribution of Conversion Rates', fontsize=14)
plt.xlabel('Conversion Rate')
plt.ylabel('Density')
plt.legend()

# Absolute difference
plt.subplot(1, 2, 2)
sns.histplot(diff_samples, kde=True, color='purple', alpha=0.5)
plt.axvline(0, color='red', linestyle='--', alpha=0.8)
plt.axvline(np.mean(diff_samples), color='purple', linestyle='--', alpha=0.8)
plt.title('Posterior Distribution of Absolute Difference (B - A)', fontsize=14)
plt.xlabel('Difference')
plt.ylabel('Density')

plt.tight_layout()
plt.savefig('bayesian_ab_test_posteriors.png', dpi=200, bbox_inches='tight')
plt.show()

#%% ROPE (Region of Practical Equivalence) analysis - Ultra Simplified
# Define ROPE boundaries (e.g., ±0.01 for absolute difference)
rope_lower = -0.01
rope_upper = 0.01

# Calculate probabilities
prob_rope = np.mean((diff_samples >= rope_lower) & (diff_samples <= rope_upper))
prob_better = np.mean(diff_samples > rope_upper)
prob_worse = np.mean(diff_samples < rope_lower)

print("\nROPE Analysis:")
print(f"Probability of practical equivalence (±{abs(rope_lower)}): {prob_rope:.2%}")
print(f"Probability that B is practically better than A: {prob_better:.2%}")
print(f"Probability that B is practically worse than A: {prob_worse:.2%}")

# Visualize ROPE analysis
plt.figure(figsize=(8, 5))
sns.histplot(diff_samples, kde=True, color='purple', alpha=0.5)
plt.axvline(rope_lower, color='red', linestyle='--', alpha=0.8, label='ROPE Lower Bound')
plt.axvline(rope_upper, color='red', linestyle='--', alpha=0.8, label='ROPE Upper Bound')
plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='No Difference')
plt.axvline(np.mean(diff_samples), color='purple', linestyle='--', alpha=0.8, label='Mean Difference')
plt.title('ROPE Analysis for Absolute Difference (B - A)', fontsize=14)
plt.xlabel('Difference')
plt.ylabel('Density')
plt.legend()
plt.savefig('bayesian_ab_test_rope.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Expected loss analysis - Ultra Simplified
# Calculate expected loss
def expected_loss(diff_samples, threshold=0):
    """
    Calculate expected loss for a given threshold.
    
    Parameters:
    -----------
    diff_samples : array-like
        Samples from the posterior distribution of the difference
    threshold : float
        Threshold for decision making
        
    Returns:
    --------
    expected_loss_a : float
        Expected loss of choosing variant A
    expected_loss_b : float
        Expected loss of choosing variant B
    """
    # Expected loss of choosing A (when B is better)
    loss_a = np.maximum(diff_samples - threshold, 0)
    expected_loss_a = np.mean(loss_a)
    
    # Expected loss of choosing B (when A is better)
    loss_b = np.maximum(threshold - diff_samples, 0)
    expected_loss_b = np.mean(loss_b)
    
    return expected_loss_a, expected_loss_b

# Calculate expected loss
expected_loss_a, expected_loss_b = expected_loss(diff_samples)

print("\nExpected Loss Analysis:")
print(f"Expected loss of choosing A: {expected_loss_a:.6f}")
print(f"Expected loss of choosing B: {expected_loss_b:.6f}")
print(f"Recommended variant: {'B' if expected_loss_b < expected_loss_a else 'A'}")

# Visualize expected loss - Simplified
plt.figure(figsize=(8, 5))
thresholds = np.linspace(-0.05, 0.05, 10)  # Minimal number of points
loss_a = []
loss_b = []

for threshold in thresholds:
    el_a, el_b = expected_loss(diff_samples, threshold)
    loss_a.append(el_a)
    loss_b.append(el_b)

plt.plot(thresholds, loss_a, 'b-', label='Expected Loss of Choosing A')
plt.plot(thresholds, loss_b, 'g-', label='Expected Loss of Choosing B')
plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='No Difference')
plt.title('Expected Loss Analysis', fontsize=14)
plt.xlabel('Threshold')
plt.ylabel('Expected Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bayesian_ab_test_expected_loss.png', dpi=200, bbox_inches='tight')
plt.show()

#%% Generate PDF report - Ultra Simplified
class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Bayesian A/B Test Analysis Report', 0, 0, 'C')
        # Line break
        self.ln(20)
    
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

# Create PDF report
pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()

# Title page
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Bayesian A/B Test Analysis Report', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
pdf.ln(20)

# Introduction
pdf.chapter_title('Introduction')
pdf.chapter_body('This report presents the results of a Bayesian A/B test analysis. Bayesian A/B testing provides a more nuanced understanding of the differences between variants compared to traditional frequentist methods. It allows us to quantify our uncertainty about the true conversion rates and make more informed decisions based on the posterior distributions.')

# Data Description
pdf.chapter_title('Data Description')
pdf.chapter_body('The analysis includes synthetic A/B test data with 200 samples for each variant. The data simulates a binary outcome (conversion) for two variants, A and B. The true conversion rates are 15% for variant A and 18% for variant B.')

# Add data visualization
pdf.ln(5)
pdf.image('ab_test_conversion_distribution.png', x=10, w=190)
pdf.ln(5)

# Frequentist Analysis
pdf.chapter_title('Frequentist Analysis (for Comparison)')
pdf.chapter_body(f'The frequentist analysis using a chi-square test resulted in a p-value of {p_value:.4f}. This suggests {"a statistically significant difference" if p_value < 0.05 else "no statistically significant difference"} between the variants at the 5% significance level.')

# Bayesian Analysis
pdf.chapter_title('Bayesian Analysis')
pdf.chapter_body(f'The Bayesian analysis provides a more nuanced understanding of the differences between variants. The posterior mean conversion rate for variant A is {np.mean(p_a_samples):.4f} (95% CI: {np.percentile(p_a_samples, 2.5):.4f} - {np.percentile(p_a_samples, 97.5):.4f}), while for variant B it is {np.mean(p_b_samples):.4f} (95% CI: {np.percentile(p_b_samples, 2.5):.4f} - {np.percentile(p_b_samples, 97.5):.4f}).')

pdf.chapter_body(f'The absolute difference between variants (B - A) is {np.mean(diff_samples):.4f} (95% CI: {np.percentile(diff_samples, 2.5):.4f} - {np.percentile(diff_samples, 97.5):.4f}). The relative improvement of B over A is {np.mean(rel_improvement_samples):.2%} (95% CI: {np.percentile(rel_improvement_samples, 2.5):.2%} - {np.percentile(rel_improvement_samples, 97.5):.2%}).')

pdf.chapter_body(f'The probability that variant B is better than variant A is {prob_b_better:.2%}.')

# Add posterior visualization
pdf.ln(5)
pdf.image('bayesian_ab_test_posteriors.png', x=10, w=190)
pdf.ln(5)

# ROPE Analysis
pdf.chapter_title('ROPE Analysis')
pdf.chapter_body(f'The Region of Practical Equivalence (ROPE) analysis defines a range of values around zero where differences are considered practically equivalent. Using a ROPE of ±{abs(rope_lower)}, we find:')
pdf.chapter_body(f'- Probability of practical equivalence: {prob_rope:.2%}')
pdf.chapter_body(f'- Probability that B is practically better than A: {prob_better:.2%}')
pdf.chapter_body(f'- Probability that B is practically worse than A: {prob_worse:.2%}')

# Add ROPE visualization
pdf.ln(5)
pdf.image('bayesian_ab_test_rope.png', x=10, w=190)
pdf.ln(5)

# Expected Loss Analysis
pdf.chapter_title('Expected Loss Analysis')
pdf.chapter_body(f'The expected loss analysis quantifies the potential cost of making the wrong decision. The expected loss of choosing variant A is {expected_loss_a:.6f}, while the expected loss of choosing variant B is {expected_loss_b:.6f}.')
pdf.chapter_body(f'Based on this analysis, the recommended variant is {"B" if expected_loss_b < expected_loss_a else "A"}.')

# Add expected loss visualization
pdf.ln(5)
pdf.image('bayesian_ab_test_expected_loss.png', x=10, w=190)
pdf.ln(5)

# Conclusion
pdf.chapter_title('Conclusion')
conclusion_text = f"""
Based on the Bayesian A/B test analysis, we can conclude that:

1. The posterior mean conversion rate for variant A is {np.mean(p_a_samples):.4f} (95% CI: {np.percentile(p_a_samples, 2.5):.4f} - {np.percentile(p_a_samples, 97.5):.4f}).
2. The posterior mean conversion rate for variant B is {np.mean(p_b_samples):.4f} (95% CI: {np.percentile(p_b_samples, 2.5):.4f} - {np.percentile(p_b_samples, 97.5):.4f}).
3. The probability that variant B is better than variant A is {prob_b_better:.2%}.
4. The expected loss of choosing variant A is {expected_loss_a:.6f}, while the expected loss of choosing variant B is {expected_loss_b:.6f}.

The recommended action is to {"implement variant B" if expected_loss_b < expected_loss_a else "keep variant A"}.
"""
pdf.chapter_body(conclusion_text)

# Save the PDF
pdf.output('bayesian_ab_test_report.pdf')

print("\nPDF report generated: bayesian_ab_test_report.pdf") 
# %%
