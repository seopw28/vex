#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, rankdata
import warnings
from fpdf import FPDF
import os
from datetime import datetime
from statsmodels.stats.power import TTestPower
warnings.filterwarnings('ignore')

#%% Generate sample data
# Let's simulate an experiment comparing the effectiveness of two different teaching methods
np.random.seed(42)

# Parameters
n_a = 30  # Number of students in group A
n_b = 30  # Number of students in group B
mean_a = 75  # True mean score for group A
mean_b = 82  # True mean score for group B
std_a = 10   # Standard deviation for group A
std_b = 12   # Standard deviation for group B

# Generate data with non-normal distribution (skewed)
scores_a = np.random.gamma(shape=2, scale=mean_a/2, size=n_a)
scores_b = np.random.gamma(shape=2, scale=mean_b/2, size=n_b)

# Create DataFrame
wilcoxon_data = pd.DataFrame({
    'Method': ['Traditional'] * n_a + ['Online'] * n_b,
    'Score': np.concatenate([scores_a, scores_b])
})

# Display sample data
print("Sample of Wilcoxon Rank-Sum Test Data:")
print(wilcoxon_data.head(10))
print("\nData Summary:")
print(wilcoxon_data.groupby('Method')['Score'].describe())

#%% Visualize data
plt.figure(figsize=(10, 6))
sns.boxplot(x='Method', y='Score', data=wilcoxon_data)
sns.swarmplot(x='Method', y='Score', data=wilcoxon_data, color='red', alpha=0.5)
plt.title('Student Scores by Teaching Method', fontsize=14)
plt.xlabel('Teaching Method')
plt.ylabel('Score')
plt.savefig('wilcoxon_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Perform Wilcoxon rank-sum test (Mann-Whitney U test)
# Extract data for each group
group_a = wilcoxon_data[wilcoxon_data['Method'] == 'Traditional']['Score']
group_b = wilcoxon_data[wilcoxon_data['Method'] == 'Online']['Score']

# Perform the test
statistic, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')

print("\nWilcoxon Rank-Sum Test Results:")
print(f"U-statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")

#%% Calculate effect size (r)
def cohens_r(statistic, n1, n2):
    """
    Calculate Cohen's r effect size for Wilcoxon rank-sum test.
    
    Parameters:
    -----------
    statistic : float
        U-statistic from Mann-Whitney U test
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2
        
    Returns:
    --------
    r : float
        Cohen's r effect size
    """
    # Calculate z-score approximation
    n = n1 + n2
    mu_u = (n1 * n2) / 2
    sigma_u = np.sqrt((n1 * n2 * (n + 1)) / 12)
    z = (statistic - mu_u) / sigma_u
    
    # Calculate Cohen's r
    r = z / np.sqrt(n)
    
    return r

# Calculate effect size
r = cohens_r(statistic, n_a, n_b)
print(f"\nEffect Size (Cohen's r): {r:.4f}")

# Interpret effect size
def interpret_cohens_r(r):
    """
    Interpret Cohen's r effect size.
    
    Parameters:
    -----------
    r : float
        Cohen's r effect size
        
    Returns:
    --------
    interpretation : str
        Text interpretation of the effect size
    """
    if abs(r) < 0.1:
        return "negligible"
    elif abs(r) < 0.3:
        return "small"
    elif abs(r) < 0.5:
        return "medium"
    else:
        return "large"

effect_size_desc = interpret_cohens_r(r)
print(f"Effect size interpretation: {effect_size_desc}")

#%% Calculate rank-biserial correlation
def rank_biserial_correlation(group1, group2):
    """
    Calculate rank-biserial correlation for Wilcoxon rank-sum test.
    
    Parameters:
    -----------
    group1 : array-like
        Data for the first group
    group2 : array-like
        Data for the second group
        
    Returns:
    --------
    rrb : float
        Rank-biserial correlation
    """
    # Combine data and assign ranks
    combined = np.concatenate([group1, group2])
    ranks = rankdata(combined)
    
    # Split ranks back into groups
    ranks1 = ranks[:len(group1)]
    ranks2 = ranks[len(group1):]
    
    # Calculate mean ranks
    mean_rank1 = np.mean(ranks1)
    mean_rank2 = np.mean(ranks2)
    
    # Calculate rank-biserial correlation
    n1 = len(group1)
    n2 = len(group2)
    rrb = (mean_rank2 - mean_rank1) / ((n1 + n2) / 2)
    
    return rrb

# Calculate rank-biserial correlation
rrb = rank_biserial_correlation(group_a, group_b)
print(f"\nRank-biserial correlation: {rrb:.4f}")

#%% Visualize rank distributions
# Create a DataFrame with ranks
combined = np.concatenate([group_a, group_b])
ranks = rankdata(combined)
ranks_a = ranks[:len(group_a)]
ranks_b = ranks[len(group_a):]

rank_data = pd.DataFrame({
    'Method': ['Traditional'] * len(ranks_a) + ['Online'] * len(ranks_b),
    'Rank': np.concatenate([ranks_a, ranks_b])
})

plt.figure(figsize=(10, 6))
sns.boxplot(x='Method', y='Rank', data=rank_data)
sns.swarmplot(x='Method', y='Rank', data=rank_data, color='red', alpha=0.5)
plt.title('Rank Distribution by Teaching Method', fontsize=14)
plt.xlabel('Teaching Method')
plt.ylabel('Rank')
plt.savefig('rank_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Power analysis for Wilcoxon rank-sum test
def power_analysis_wilcoxon(effect_size, alpha=0.05, power=0.8):
    """
    Perform power analysis for Wilcoxon rank-sum test.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's r effect size
    alpha : float
        Significance level (default: 0.05)
    power : float
        Desired power (default: 0.8)
        
    Returns:
    --------
    n_per_group : int
        Required sample size per group
    """
    # For non-parametric tests, we can use the t-test power analysis as an approximation
    # and then add a 15% buffer to account for the loss of power
    analysis = TTestPower()
    
    # Calculate required sample size
    n_per_group = analysis.solve_power(effect_size=effect_size, 
                                      alpha=alpha, 
                                      power=power)
    
    # Add 15% buffer for non-parametric test
    n_per_group = int(np.ceil(n_per_group * 1.15))
    
    return n_per_group

# Calculate required sample size for 80% power
required_n = power_analysis_wilcoxon(r)
print(f"\nPower Analysis:")
print(f"Required sample size per group for 80% power: {required_n}")

# Calculate power for a range of sample sizes
n_range = np.arange(10, 100, 5)
power_values = []

for n in n_range:
    # Approximate power using t-test power
    power_value = TTestPower().power(effect_size=r, 
                                     nobs=n, 
                                     alpha=0.05)
    # Adjust for non-parametric test (reduce power by 15%)
    power_value = power_value * 0.85
    power_values.append(power_value)

# Visualize power curve
plt.figure(figsize=(10, 6))
plt.plot(n_range, power_values, 'b-', linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')
plt.axvline(x=required_n, color='g', linestyle='--', label=f'Required n={required_n}')
plt.title('Power Curve for Wilcoxon Rank-Sum Test', fontsize=14)
plt.xlabel('Sample Size per Group')
plt.ylabel('Power')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('power_curve.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Additional analysis: Bootstrap confidence interval for effect size
def bootstrap_effect_size_ci(group1, group2, n_bootstrap=10000, conf_level=0.95):
    """
    Calculate bootstrap confidence interval for effect size.
    
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
    bootstrap_rs : array
        Array of bootstrap effect size values
    """
    bootstrap_rs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        
        # Calculate statistic for this bootstrap sample
        stat, _ = mannwhitneyu(sample1, sample2, alternative='two-sided')
        
        # Calculate effect size
        r_boot = cohens_r(stat, len(sample1), len(sample2))
        bootstrap_rs[i] = r_boot
    
    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_rs, ((1 - conf_level) / 2) * 100)
    ci_upper = np.percentile(bootstrap_rs, (1 - ((1 - conf_level) / 2)) * 100)
    
    return ci_lower, ci_upper, bootstrap_rs

# Calculate bootstrap confidence interval
ci_lower, ci_upper, bootstrap_rs = bootstrap_effect_size_ci(group_a, group_b)

print("\nBootstrap Confidence Interval for Effect Size:")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_rs, kde=True)
plt.axvline(x=r, color='red', linestyle='--', label=f"Observed r = {r:.4f}")
plt.axvline(x=ci_lower, color='blue', linestyle=':', label='95% CI Lower')
plt.axvline(x=ci_upper, color='blue', linestyle=':', label='95% CI Upper')
plt.title("Bootstrap Distribution of Effect Size", fontsize=14)
plt.xlabel("Cohen's r")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bootstrap_effect_size.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Conclusion
print("\nConclusion:")
print(f"1. The Wilcoxon rank-sum test shows a {'statistically significant' if p_value < 0.05 else 'not statistically significant'} difference between teaching methods (p = {p_value:.4f})")
print(f"2. The effect size is {effect_size_desc} (Cohen's r = {r:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
print(f"3. The rank-biserial correlation is {rrb:.4f}, indicating the direction and magnitude of the difference")
print(f"4. To achieve 80% power, {required_n} participants per group would be needed")
print(f"5. Based on the current sample size (n={n_a} per group), the approximate power is {power_values[0]:.4f}")

#%% Generate PDF report
class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Wilcoxon Rank-Sum Test Report', 0, 0, 'C')
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
pdf.cell(0, 10, 'Wilcoxon Rank-Sum Test Analysis Report', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
pdf.ln(20)

# Introduction
pdf.chapter_title('Introduction')
pdf.chapter_body('This report presents the results of a Wilcoxon rank-sum test (also known as the Mann-Whitney U test) comparing two independent groups. The test is a non-parametric alternative to the independent t-test, used when the assumption of normality is violated or when analyzing ordinal data.')

# Data Description
pdf.chapter_title('Data Description')
pdf.chapter_body(f'The analysis includes data from {n_a + n_b} participants, divided into two groups: Traditional (n={n_a}) and Online (n={n_b}). The data represents student scores after being taught using different teaching methods.')

# Add data summary table
pdf.ln(5)
pdf.set_font('Arial', 'B', 11)
pdf.cell(47.5, 7, 'Method', 1, 0, 'C')
pdf.cell(47.5, 7, 'Mean', 1, 0, 'C')
pdf.cell(47.5, 7, 'Median', 1, 0, 'C')
pdf.cell(47.5, 7, 'Std Dev', 1, 1, 'C')

pdf.set_font('Arial', '', 11)
pdf.cell(47.5, 7, 'Traditional', 1, 0, 'C')
pdf.cell(47.5, 7, f'{np.mean(group_a):.2f}', 1, 0, 'C')
pdf.cell(47.5, 7, f'{np.median(group_a):.2f}', 1, 0, 'C')
pdf.cell(47.5, 7, f'{np.std(group_a):.2f}', 1, 1, 'C')

pdf.cell(47.5, 7, 'Online', 1, 0, 'C')
pdf.cell(47.5, 7, f'{np.mean(group_b):.2f}', 1, 0, 'C')
pdf.cell(47.5, 7, f'{np.median(group_b):.2f}', 1, 0, 'C')
pdf.cell(47.5, 7, f'{np.std(group_b):.2f}', 1, 1, 'C')

# Add boxplot image
pdf.ln(10)
pdf.image('wilcoxon_boxplot.png', x=10, w=190)

# Test Results
pdf.add_page()
pdf.chapter_title('Test Results')
pdf.chapter_body(f'The Wilcoxon rank-sum test was performed to compare the scores between the Traditional and Online teaching methods. The results indicate a {"statistically significant" if p_value < 0.05 else "not statistically significant"} difference between the groups (U = {statistic:.2f}, p = {p_value:.4f}).')

# Effect Size
pdf.chapter_title('Effect Size')
pdf.chapter_body(f'The effect size was calculated using Cohen\'s r, which is appropriate for non-parametric tests. The effect size is {effect_size_desc} (r = {r:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). This indicates the magnitude of the difference between the two teaching methods.')

# Add effect size visualization
pdf.ln(5)
pdf.image('bootstrap_effect_size.png', x=10, w=190)

# Rank-biserial Correlation
pdf.chapter_title('Rank-biserial Correlation')
pdf.chapter_body(f'The rank-biserial correlation (r_rb = {rrb:.4f}) indicates the direction and magnitude of the difference between the two groups. A positive value suggests that the Online method is associated with higher ranks (better scores) compared to the Traditional method.')

# Power Analysis
pdf.add_page()
pdf.chapter_title('Power Analysis')
pdf.chapter_body(f'A power analysis was conducted to determine the sample size required to achieve 80% power with the observed effect size. The results indicate that {required_n} participants per group would be needed to achieve this level of power. With the current sample size (n={n_a} per group), the approximate power is {power_values[0]:.4f}.')

# Add power curve image
pdf.ln(5)
pdf.image('power_curve.png', x=10, w=190)

# Conclusion
pdf.chapter_title('Conclusion')
conclusion_text = f"""
Based on the Wilcoxon rank-sum test, there is a {"statistically significant" if p_value < 0.05 else "not statistically significant"} difference in student scores between the Traditional and Online teaching methods (p = {p_value:.4f}). 

The effect size is {effect_size_desc} (Cohen's r = {r:.4f}), indicating the magnitude of this difference. The rank-biserial correlation (r_rb = {rrb:.4f}) shows the direction of the difference, suggesting that the {"Online" if rrb > 0 else "Traditional"} method is associated with {"higher" if rrb > 0 else "lower"} scores.

To achieve 80% power with the observed effect size, {required_n} participants per group would be needed. The current study has approximately {power_values[0]:.4f} power to detect the observed effect size.
"""
pdf.chapter_body(conclusion_text)

# Save the PDF
pdf.output('wilcoxon_rank_sum_report.pdf')

print("\nPDF report generated: wilcoxon_rank_sum_report.pdf") 
# %%
