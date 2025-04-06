#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact
import warnings
from fpdf import FPDF
import os
from datetime import datetime
warnings.filterwarnings('ignore')

#%% Generate sample data
# Let's simulate a clinical trial comparing two treatments for a rare disease
np.random.seed(42)

# Create a 2x2 contingency table
# Treatment A vs. Treatment B
# Success vs. Failure

# Parameters
n_a = 50  # Number of patients in treatment A
n_b = 50  # Number of patients in treatment B
p_a = 0.3  # True success rate for treatment A
p_b = 0.5  # True success rate for treatment B

# Generate data
success_a = np.random.binomial(n_a, p_a)
success_b = np.random.binomial(n_b, p_b)
failure_a = n_a - success_a
failure_b = n_b - success_b

# Create contingency table
contingency_table = np.array([
    [success_a, failure_a],
    [success_b, failure_b]
])

# Create DataFrame for visualization
fisher_data = pd.DataFrame({
    'Treatment': ['A'] * n_a + ['B'] * n_b,
    'Outcome': ['Success'] * success_a + ['Failure'] * failure_a + 
               ['Success'] * success_b + ['Failure'] * failure_b
})

# Display sample data
print("Sample of Fisher's Exact Test Data:")
print(fisher_data.head(10))
print("\nContingency Table:")
print(pd.DataFrame(
    contingency_table, 
    index=['Treatment A', 'Treatment B'],
    columns=['Success', 'Failure']
))

#%% Visualize data
# Create a stacked bar chart
plt.figure(figsize=(10, 6))
contingency_df = pd.DataFrame(
    contingency_table, 
    index=['Treatment A', 'Treatment B'],
    columns=['Success', 'Failure']
)
contingency_df.plot(kind='bar', stacked=True, color=['green', 'red'])
plt.title('Treatment Outcomes', fontsize=14)
plt.xlabel('Treatment')
plt.ylabel('Number of Patients')
plt.legend(title='Outcome')
plt.grid(axis='y', alpha=0.3)
plt.savefig('fisher_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=['Success', 'Failure'],
            yticklabels=['Treatment A', 'Treatment B'])
plt.title('Contingency Table Heatmap', fontsize=14)
plt.savefig('fisher_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Perform Fisher's Exact Test
# Perform the test
odds_ratio, p_value = fisher_exact(contingency_table)

print("\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")

#%% Calculate effect size (phi coefficient)
def phi_coefficient(contingency_table):
    """
    Calculate phi coefficient for 2x2 contingency table.
    
    Parameters:
    -----------
    contingency_table : array-like
        2x2 contingency table
        
    Returns:
    --------
    phi : float
        Phi coefficient
    """
    a, b = contingency_table[0, 0], contingency_table[0, 1]
    c, d = contingency_table[1, 0], contingency_table[1, 1]
    
    n = a + b + c + d
    
    phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    
    return phi

# Calculate phi coefficient
phi = phi_coefficient(contingency_table)
print(f"\nEffect Size (Phi Coefficient): {phi:.4f}")

# Interpret effect size
def interpret_phi(phi):
    """
    Interpret phi coefficient effect size.
    
    Parameters:
    -----------
    phi : float
        Phi coefficient
        
    Returns:
    --------
    interpretation : str
        Text interpretation of the effect size
    """
    if abs(phi) < 0.1:
        return "negligible"
    elif abs(phi) < 0.3:
        return "small"
    elif abs(phi) < 0.5:
        return "medium"
    else:
        return "large"

effect_size_desc = interpret_phi(phi)
print(f"Effect size interpretation: {effect_size_desc}")

#%% Calculate relative risk
def relative_risk(contingency_table):
    """
    Calculate relative risk for 2x2 contingency table.
    
    Parameters:
    -----------
    contingency_table : array-like
        2x2 contingency table
        
    Returns:
    --------
    rr : float
        Relative risk
    ci_lower : float
        Lower bound of 95% confidence interval
    ci_upper : float
        Upper bound of 95% confidence interval
    """
    a, b = contingency_table[0, 0], contingency_table[0, 1]
    c, d = contingency_table[1, 0], contingency_table[1, 1]
    
    # Calculate risk for each group
    risk_a = a / (a + b)
    risk_b = c / (c + d)
    
    # Calculate relative risk
    rr = risk_a / risk_b
    
    # Calculate standard error of log(RR)
    se_log_rr = np.sqrt(1/a + 1/c - 1/(a+b) - 1/(c+d))
    
    # Calculate 95% confidence interval
    ci_lower = np.exp(np.log(rr) - 1.96 * se_log_rr)
    ci_upper = np.exp(np.log(rr) + 1.96 * se_log_rr)
    
    return rr, ci_lower, ci_upper

# Calculate relative risk
rr, rr_ci_lower, rr_ci_upper = relative_risk(contingency_table)
print(f"\nRelative Risk: {rr:.4f} (95% CI: [{rr_ci_lower:.4f}, {rr_ci_upper:.4f}])")

#%% Visualize odds ratio and relative risk
plt.figure(figsize=(12, 6))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Odds Ratio plot
ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax1.errorbar(odds_ratio, 0, xerr=[[odds_ratio - np.exp(np.log(odds_ratio) - 1.96 * np.sqrt(1/contingency_table[0,0] + 1/contingency_table[0,1] + 1/contingency_table[1,0] + 1/contingency_table[1,1]))], 
             [np.exp(np.log(odds_ratio) + 1.96 * np.sqrt(1/contingency_table[0,0] + 1/contingency_table[0,1] + 1/contingency_table[1,0] + 1/contingency_table[1,1])) - odds_ratio]], 
             fmt='o', color='blue', capsize=5, markersize=10)
ax1.set_xscale('log')
ax1.set_xlabel('Odds Ratio (log scale)')
ax1.set_title('Odds Ratio with 95% CI', fontsize=14)
ax1.grid(True, alpha=0.3)

# Relative Risk plot
ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax2.errorbar(rr, 0, xerr=[[rr - rr_ci_lower], [rr_ci_upper - rr]], 
             fmt='o', color='green', capsize=5, markersize=10)
ax2.set_xscale('log')
ax2.set_xlabel('Relative Risk (log scale)')
ax2.set_title('Relative Risk with 95% CI', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('effect_measures.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Power analysis for Fisher's Exact Test
def power_analysis_fisher(p1, p2, n1, n2, alpha=0.05):
    """
    Perform power analysis for Fisher's Exact Test.
    
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
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    power : float
        Statistical power
    """
    # For Fisher's Exact Test, we can use chi-square power analysis as an approximation
    from statsmodels.stats.power import TTestPower
    
    # Calculate effect size (phi)
    a = int(n1 * p1)
    b = int(n1 * (1 - p1))
    c = int(n2 * p2)
    d = int(n2 * (1 - p2))
    
    phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    
    # Calculate power
    analysis = TTestPower()
    power = analysis.power(effect_size=phi, 
                          nobs=min(n1, n2), 
                          alpha=alpha)
    
    return power

# Calculate power for current sample size
current_power = power_analysis_fisher(p_a, p_b, n_a, n_b)
print(f"\nPower Analysis:")
print(f"Current power: {current_power:.4f}")

# Calculate power for a range of sample sizes
n_range = np.arange(10, 200, 10)
power_values = []

for n in n_range:
    power_value = power_analysis_fisher(p_a, p_b, n, n)
    power_values.append(power_value)

# Find required sample size for 80% power
required_n = None
for i, power in enumerate(power_values):
    if power >= 0.8:
        required_n = n_range[i]
        break

if required_n is None:
    required_n = n_range[-1]
    print(f"Note: Even with {required_n} participants per group, power is still below 0.8")

print(f"Required sample size per group for 80% power: {required_n}")

# Visualize power curve
plt.figure(figsize=(10, 6))
plt.plot(n_range, power_values, 'b-', linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')
if required_n is not None:
    plt.axvline(x=required_n, color='g', linestyle='--', label=f'Required n={required_n}')
plt.title('Power Curve for Fisher\'s Exact Test', fontsize=14)
plt.xlabel('Sample Size per Group')
plt.ylabel('Power')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('power_curve.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Additional analysis: Bootstrap confidence interval for odds ratio
def bootstrap_odds_ratio_ci(contingency_table, n_bootstrap=10000, conf_level=0.95):
    """
    Calculate bootstrap confidence interval for odds ratio.
    
    Parameters:
    -----------
    contingency_table : array-like
        2x2 contingency table
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
    bootstrap_ors : array
        Array of bootstrap odds ratio values
    """
    a, b = contingency_table[0, 0], contingency_table[0, 1]
    c, d = contingency_table[1, 0], contingency_table[1, 1]
    
    # Create data for bootstrapping
    group1 = np.array([1] * a + [0] * b)
    group2 = np.array([1] * c + [0] * d)
    
    bootstrap_ors = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        
        # Calculate counts for this bootstrap sample
        a_boot = np.sum(sample1 == 1)
        b_boot = np.sum(sample1 == 0)
        c_boot = np.sum(sample2 == 1)
        d_boot = np.sum(sample2 == 0)
        
        # Calculate odds ratio
        if a_boot * d_boot == 0 or b_boot * c_boot == 0:
            bootstrap_ors[i] = np.nan
        else:
            bootstrap_ors[i] = (a_boot * d_boot) / (b_boot * c_boot)
    
    # Remove NaN values
    bootstrap_ors = bootstrap_ors[~np.isnan(bootstrap_ors)]
    
    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_ors, ((1 - conf_level) / 2) * 100)
    ci_upper = np.percentile(bootstrap_ors, (1 - ((1 - conf_level) / 2)) * 100)
    
    return ci_lower, ci_upper, bootstrap_ors

# Calculate bootstrap confidence interval
or_ci_lower, or_ci_upper, bootstrap_ors = bootstrap_odds_ratio_ci(contingency_table)

print("\nBootstrap Confidence Interval for Odds Ratio:")
print(f"95% CI: [{or_ci_lower:.4f}, {or_ci_upper:.4f}]")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_ors, kde=True)
plt.axvline(x=odds_ratio, color='red', linestyle='--', label=f"Observed OR = {odds_ratio:.4f}")
plt.axvline(x=or_ci_lower, color='blue', linestyle=':', label='95% CI Lower')
plt.axvline(x=or_ci_upper, color='blue', linestyle=':', label='95% CI Upper')
plt.title("Bootstrap Distribution of Odds Ratio", fontsize=14)
plt.xlabel("Odds Ratio")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bootstrap_odds_ratio.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Conclusion
print("\nConclusion:")
print(f"1. Fisher's Exact Test shows a {'statistically significant' if p_value < 0.05 else 'not statistically significant'} association between treatment and outcome (p = {p_value:.4f})")
print(f"2. The effect size is {effect_size_desc} (Phi = {phi:.4f})")
print(f"3. The odds ratio is {odds_ratio:.4f} (95% CI: [{or_ci_lower:.4f}, {or_ci_upper:.4f}])")
print(f"4. The relative risk is {rr:.4f} (95% CI: [{rr_ci_lower:.4f}, {rr_ci_upper:.4f}])")
print(f"5. To achieve 80% power, {required_n} participants per group would be needed")
print(f"6. Based on the current sample size (n={n_a} per group), the power is {current_power:.4f}")

#%% Generate PDF report
class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Fisher\'s Exact Test Report', 0, 0, 'C')
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
pdf.cell(0, 10, 'Fisher\'s Exact Test Analysis Report', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
pdf.ln(20)

# Introduction
pdf.chapter_title('Introduction')
pdf.chapter_body('This report presents the results of a Fisher\'s Exact Test comparing two independent groups. The test is a non-parametric statistical test used to determine if there is a significant association between two categorical variables in a 2x2 contingency table. It is particularly useful when sample sizes are small or when expected cell counts are less than 5.')

# Data Description
pdf.chapter_title('Data Description')
pdf.chapter_body(f'The analysis includes data from {n_a + n_b} participants, divided into two groups: Treatment A (n={n_a}) and Treatment B (n={n_b}). The data represents success or failure outcomes after receiving different treatments.')

# Add contingency table
pdf.ln(5)
pdf.set_font('Arial', 'B', 11)
pdf.cell(47.5, 7, '', 1, 0, 'C')
pdf.cell(47.5, 7, 'Success', 1, 0, 'C')
pdf.cell(47.5, 7, 'Failure', 1, 1, 'C')

pdf.set_font('Arial', '', 11)
pdf.cell(47.5, 7, 'Treatment A', 1, 0, 'C')
pdf.cell(47.5, 7, f'{contingency_table[0, 0]}', 1, 0, 'C')
pdf.cell(47.5, 7, f'{contingency_table[0, 1]}', 1, 1, 'C')

pdf.cell(47.5, 7, 'Treatment B', 1, 0, 'C')
pdf.cell(47.5, 7, f'{contingency_table[1, 0]}', 1, 0, 'C')
pdf.cell(47.5, 7, f'{contingency_table[1, 1]}', 1, 1, 'C')

# Add visualizations
pdf.ln(10)
pdf.image('fisher_barplot.png', x=10, w=190)
pdf.ln(5)
pdf.image('fisher_heatmap.png', x=10, w=190)

# Test Results
pdf.add_page()
pdf.chapter_title('Test Results')
pdf.chapter_body(f'Fisher\'s Exact Test was performed to compare the success rates between Treatment A and Treatment B. The results indicate a {"statistically significant" if p_value < 0.05 else "not statistically significant"} association between treatment and outcome (p = {p_value:.4f}).')

# Effect Size
pdf.chapter_title('Effect Size')
pdf.chapter_body(f'The effect size was calculated using the Phi coefficient, which is appropriate for 2x2 contingency tables. The effect size is {effect_size_desc} (Phi = {phi:.4f}). This indicates the strength of the association between treatment and outcome.')

# Odds Ratio and Relative Risk
pdf.chapter_title('Odds Ratio and Relative Risk')
pdf.chapter_body(f'The odds ratio (OR = {odds_ratio:.4f}, 95% CI: [{or_ci_lower:.4f}, {or_ci_upper:.4f}]) indicates the odds of success in Treatment A compared to Treatment B. A value greater than 1 suggests that Treatment A is associated with higher odds of success.')
pdf.chapter_body(f'The relative risk (RR = {rr:.4f}, 95% CI: [{rr_ci_lower:.4f}, {rr_ci_upper:.4f}]) indicates the risk of success in Treatment A compared to Treatment B. A value greater than 1 suggests that Treatment A is associated with a higher risk of success.')

# Add effect measures visualization
pdf.ln(5)
pdf.image('effect_measures.png', x=10, w=190)

# Power Analysis
pdf.add_page()
pdf.chapter_title('Power Analysis')
pdf.chapter_body(f'A power analysis was conducted to determine the sample size required to achieve 80% power with the observed effect size. The results indicate that {required_n} participants per group would be needed to achieve this level of power. With the current sample size (n={n_a} per group), the power is {current_power:.4f}.')

# Add power curve image
pdf.ln(5)
pdf.image('power_curve.png', x=10, w=190)

# Bootstrap Analysis
pdf.chapter_title('Bootstrap Analysis')
pdf.chapter_body(f'A bootstrap analysis was conducted to estimate the confidence interval for the odds ratio. The results indicate that the 95% confidence interval for the odds ratio is [{or_ci_lower:.4f}, {or_ci_upper:.4f}]. This provides a more robust estimate of the uncertainty in the odds ratio compared to the asymptotic confidence interval.')

# Add bootstrap visualization
pdf.ln(5)
pdf.image('bootstrap_odds_ratio.png', x=10, w=190)

# Conclusion
pdf.chapter_title('Conclusion')
conclusion_text = f"""
Based on Fisher's Exact Test, there is a {"statistically significant" if p_value < 0.05 else "not statistically significant"} association between treatment and outcome (p = {p_value:.4f}). 

The effect size is {effect_size_desc} (Phi = {phi:.4f}), indicating the strength of this association. The odds ratio (OR = {odds_ratio:.4f}, 95% CI: [{or_ci_lower:.4f}, {or_ci_upper:.4f}]) and relative risk (RR = {rr:.4f}, 95% CI: [{rr_ci_lower:.4f}, {rr_ci_upper:.4f}]) provide additional measures of the treatment effect.

To achieve 80% power with the observed effect size, {required_n} participants per group would be needed. The current study has {current_power:.4f} power to detect the observed effect size.
"""
pdf.chapter_body(conclusion_text)

# Save the PDF
pdf.output('fishers_exact_test_report.pdf')

print("\nPDF report generated: fishers_exact_test_report.pdf")
