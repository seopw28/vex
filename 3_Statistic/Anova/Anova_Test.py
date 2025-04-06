#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import warnings
warnings.filterwarnings('ignore')

#%% Generate sample data for one-way ANOVA
# Let's simulate an experiment comparing the effectiveness of three different teaching methods
np.random.seed(42)

# Parameters
n_per_group = 30  # Number of students per teaching method
methods = ['Traditional', 'Online', 'Hybrid']
mean_scores = [75, 82, 78]  # True mean scores for each method
std_scores = [8, 7, 9]      # Standard deviation for each method

# Generate data
data = []
for i, method in enumerate(methods):
    scores = np.random.normal(mean_scores[i], std_scores[i], n_per_group)
    for score in scores:
        data.append({'Method': method, 'Score': score})

# Create DataFrame
one_way_data = pd.DataFrame(data)

# Display sample data
print("Sample of One-Way ANOVA Data:")
print(one_way_data.head(10))
print("\nData Summary:")
print(one_way_data.groupby('Method')['Score'].describe())

#%% Visualize one-way ANOVA data
plt.figure(figsize=(10, 6))
sns.boxplot(x='Method', y='Score', data=one_way_data)
sns.swarmplot(x='Method', y='Score', data=one_way_data, color='red', alpha=0.5)
plt.title('Student Scores by Teaching Method', fontsize=14)
plt.xlabel('Teaching Method')
plt.ylabel('Score')
plt.savefig('one_way_anova_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Perform one-way ANOVA
# Using scipy.stats
f_stat, p_value = stats.f_oneway(
    one_way_data[one_way_data['Method'] == 'Traditional']['Score'],
    one_way_data[one_way_data['Method'] == 'Online']['Score'],
    one_way_data[one_way_data['Method'] == 'Hybrid']['Score']
)

print("\nOne-Way ANOVA Results (scipy.stats):")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# Using statsmodels
model = sm.formula.ols('Score ~ Method', data=one_way_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nOne-Way ANOVA Results (statsmodels):")
print(anova_table)

#%% Post-hoc tests for one-way ANOVA
# Tukey's HSD test
# Since pairwise_tukeyhsd is not available in current statsmodels, we'll implement our own version
from itertools import combinations

# Get unique groups
groups = one_way_data['Method'].unique()
n_groups = len(groups)
n_total = len(one_way_data)

# Calculate group means and sizes
group_means = {}
group_sizes = {}
for group in groups:
    group_data = one_way_data[one_way_data['Method'] == group]['Score']
    group_means[group] = group_data.mean()
    group_sizes[group] = len(group_data)

# Calculate MSE (Mean Square Error) from the ANOVA model
mse = model.mse_resid

# Calculate critical value for Tukey's test
# Using studentized range distribution
from scipy.stats import studentized_range
alpha = 0.05
df = n_total - n_groups  # degrees of freedom for error
q_critical = studentized_range.ppf(1 - alpha, n_groups, df)

# Perform pairwise comparisons
tukey_results = []
for group1, group2 in combinations(groups, 2):
    # Calculate test statistic
    mean_diff = abs(group_means[group1] - group_means[group2])
    se = np.sqrt(mse * (1/group_sizes[group1] + 1/group_sizes[group2]))
    q_stat = mean_diff / se
    
    # Calculate p-value
    p_value = 1 - studentized_range.cdf(q_stat, n_groups, df)
    
    # Determine if difference is significant
    significant = q_stat > q_critical
    
    tukey_results.append({
        'Group1': group1,
        'Group2': group2,
        'Mean Diff': mean_diff,
        'SE': se,
        'q-statistic': q_stat,
        'p-value': p_value,
        'Significant': significant
    })

tukey_df = pd.DataFrame(tukey_results)
print("\nTukey's HSD Post-hoc Test:")
print(tukey_df)

# Visualize Tukey's HSD
plt.figure(figsize=(10, 6))
# Create a bar plot of group means with error bars
group_data = [one_way_data[one_way_data['Method'] == group]['Score'] for group in groups]
plt.boxplot(group_data, labels=groups)
plt.title("Group Means with Tukey's HSD", fontsize=14)
plt.xlabel('Teaching Method')
plt.ylabel('Score')

# Add significance bars for significant differences
y_max = max([max(data) for data in group_data])
y_min = min([min(data) for data in group_data])
y_range = y_max - y_min
y_offset = y_range * 0.05

for i, result in enumerate(tukey_df[tukey_df['Significant']].itertuples()):
    group1_idx = list(groups).index(result.Group1)
    group2_idx = list(groups).index(result.Group2)
    
    # Draw a line between significant groups
    plt.plot([group1_idx + 1, group2_idx + 1], 
             [y_max + y_offset + i*y_offset, y_max + y_offset + i*y_offset], 
             'k-', linewidth=1)
    
    # Add asterisk for significance
    plt.text((group1_idx + group2_idx + 2)/2, 
             y_max + y_offset + i*y_offset + y_offset/2, 
             '*', ha='center', va='bottom')

plt.savefig('tukey_hsd.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Generate sample data for two-way ANOVA
# Let's simulate an experiment comparing the effectiveness of teaching methods and study time
np.random.seed(42)

# Parameters
methods = ['Traditional', 'Online', 'Hybrid']
study_times = ['Low', 'Medium', 'High']
n_per_cell = 20  # Number of students per method x study time combination

# True mean scores for each combination
mean_scores = {
    'Traditional': {'Low': 70, 'Medium': 75, 'High': 80},
    'Online': {'Low': 75, 'Medium': 82, 'High': 88},
    'Hybrid': {'Low': 72, 'Medium': 78, 'High': 85}
}

# Standard deviation for each combination
std_scores = {
    'Traditional': {'Low': 8, 'Medium': 7, 'High': 6},
    'Online': {'Low': 7, 'Medium': 6, 'High': 5},
    'Hybrid': {'Low': 8, 'Medium': 7, 'High': 6}
}

# Generate data
data = []
for method in methods:
    for study_time in study_times:
        scores = np.random.normal(mean_scores[method][study_time], 
                                 std_scores[method][study_time], 
                                 n_per_cell)
        for score in scores:
            data.append({
                'Method': method,
                'StudyTime': study_time,
                'Score': score
            })

# Create DataFrame
two_way_data = pd.DataFrame(data)

# Display sample data
print("\nSample of Two-Way ANOVA Data:")
print(two_way_data.head(10))
print("\nData Summary:")
print(two_way_data.groupby(['Method', 'StudyTime'])['Score'].describe())

#%% Visualize two-way ANOVA data
plt.figure(figsize=(12, 8))
sns.boxplot(x='Method', y='Score', hue='StudyTime', data=two_way_data)
plt.title('Student Scores by Teaching Method and Study Time', fontsize=14)
plt.xlabel('Teaching Method')
plt.ylabel('Score')
plt.savefig('two_way_anova_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Interaction plot
plt.figure(figsize=(12, 8))
sns.pointplot(x='Method', y='Score', hue='StudyTime', data=two_way_data, 
              markers=['o', 's', '^'], linestyles=['-', '--', '-.'])
plt.title('Interaction Plot: Teaching Method × Study Time', fontsize=14)
plt.xlabel('Teaching Method')
plt.ylabel('Mean Score')
plt.savefig('interaction_plot.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Perform two-way ANOVA
# Using statsmodels
model = sm.formula.ols('Score ~ Method + StudyTime + Method:StudyTime', data=two_way_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nTwo-Way ANOVA Results:")
print(anova_table)

#%% Generate sample data for repeated measures ANOVA
# Let's simulate an experiment measuring student performance before and after a training program
np.random.seed(42)

# Parameters
n_subjects = 30  # Number of students
time_points = ['Pre', 'Post', 'Follow-up']
n_time_points = len(time_points)

# Generate baseline scores
baseline_scores = np.random.normal(70, 10, n_subjects)

# Generate improvement factors for each time point
improvement_factors = {
    'Pre': 0,      # No improvement at baseline
    'Post': 8,     # Immediate improvement after training
    'Follow-up': 5  # Some retention after time
}

# Generate data
data = []
for subject in range(n_subjects):
    for time in time_points:
        improvement = improvement_factors[time]
        # Add some random variation to improvement
        random_improvement = np.random.normal(improvement, 2)
        score = baseline_scores[subject] + random_improvement
        # Ensure score is within reasonable range
        score = max(0, min(100, score))
        data.append({
            'Subject': f'Subject_{subject+1}',
            'Time': time,
            'Score': score
        })

# Create DataFrame
repeated_measures_data = pd.DataFrame(data)

# Display sample data
print("\nSample of Repeated Measures ANOVA Data:")
print(repeated_measures_data.head(10))
print("\nData Summary:")
print(repeated_measures_data.groupby('Time')['Score'].describe())

#%% Visualize repeated measures ANOVA data
plt.figure(figsize=(10, 6))
sns.boxplot(x='Time', y='Score', data=repeated_measures_data)
sns.swarmplot(x='Time', y='Score', data=repeated_measures_data, color='red', alpha=0.5)
plt.title('Student Scores Across Time Points', fontsize=14)
plt.xlabel('Time Point')
plt.ylabel('Score')
plt.savefig('repeated_measures_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Individual subject lines
plt.figure(figsize=(12, 8))
for subject in repeated_measures_data['Subject'].unique()[:10]:  # Show first 10 subjects
    subject_data = repeated_measures_data[repeated_measures_data['Subject'] == subject]
    plt.plot(subject_data['Time'], subject_data['Score'], marker='o', label=subject)
plt.title('Individual Subject Scores Across Time Points', fontsize=14)
plt.xlabel('Time Point')
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('individual_subject_lines.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Perform repeated measures ANOVA
# Using statsmodels
aovrm = AnovaRM(data=repeated_measures_data, depvar='Score', subject='Subject', 
                within=['Time']).fit()
print("\nRepeated Measures ANOVA Results:")
print(aovrm.anova_table)

#%% Post-hoc tests for repeated measures ANOVA
# Pairwise t-tests with Bonferroni correction
from itertools import combinations

time_points = repeated_measures_data['Time'].unique()
pairs = list(combinations(time_points, 2))
results = []

for t1, t2 in pairs:
    t1_data = repeated_measures_data[repeated_measures_data['Time'] == t1]['Score']
    t2_data = repeated_measures_data[repeated_measures_data['Time'] == t2]['Score']
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(t1_data, t2_data)
    
    results.append({
        'Comparison': f'{t1} vs {t2}',
        't-statistic': t_stat,
        'p-value': p_val,
        'p-value (Bonferroni)': min(p_val * len(pairs), 1.0)  # Bonferroni correction
    })

post_hoc_results = pd.DataFrame(results)
print("\nPost-hoc Paired t-tests with Bonferroni Correction:")
print(post_hoc_results)

#%% Effect size calculations
# For one-way ANOVA: Eta-squared
# Check the actual column names in the ANOVA table
print("\nANOVA Table Column Names:")
print(anova_table.columns)

# Use the correct column names based on the actual output
# For one-way ANOVA
if 'sum_sq' in anova_table.columns:
    ss_between = anova_table.loc['Method', 'sum_sq']
    ss_total = anova_table['sum_sq'].sum()
else:
    # Try alternative column names that might be used in newer versions
    if 'ss' in anova_table.columns:
        ss_between = anova_table.loc['Method', 'ss']
        ss_total = anova_table['ss'].sum()
    elif 'SS' in anova_table.columns:
        ss_between = anova_table.loc['Method', 'SS']
        ss_total = anova_table['SS'].sum()
    else:
        # If we can't find the right column, calculate it manually
        print("Warning: Could not find sum of squares column. Calculating manually.")
        # Calculate total sum of squares
        grand_mean = one_way_data['Score'].mean()
        ss_total = sum((one_way_data['Score'] - grand_mean)**2)
        
        # Calculate between-group sum of squares
        ss_between = 0
        for method in one_way_data['Method'].unique():
            group_data = one_way_data[one_way_data['Method'] == method]['Score']
            group_mean = group_data.mean()
            ss_between += len(group_data) * (group_mean - grand_mean)**2

eta_squared = ss_between / ss_total

print("\nEffect Size for One-Way ANOVA:")
print(f"Eta-squared: {eta_squared:.4f}")

# For repeated measures ANOVA: Partial Eta-squared
# Check the actual column names in the repeated measures ANOVA table
print("\nRepeated Measures ANOVA Table Column Names:")
print(aovrm.anova_table.columns)

# Use the correct column names based on the actual output
if 'sum_sq' in aovrm.anova_table.columns:
    ss_time = aovrm.anova_table.loc['Time', 'sum_sq']
    ss_error = aovrm.anova_table.loc['Time', 'sum_sq'] + aovrm.anova_table.loc['Error', 'sum_sq']
else:
    # Try alternative column names
    if 'ss' in aovrm.anova_table.columns:
        ss_time = aovrm.anova_table.loc['Time', 'ss']
        ss_error = aovrm.anova_table.loc['Time', 'ss'] + aovrm.anova_table.loc['Error', 'ss']
    elif 'SS' in aovrm.anova_table.columns:
        ss_time = aovrm.anova_table.loc['Time', 'SS']
        ss_error = aovrm.anova_table.loc['Time', 'SS'] + aovrm.anova_table.loc['Error', 'SS']
    else:
        # If we can't find the right column, calculate it manually
        print("Warning: Could not find sum of squares column. Calculating manually.")
        # This is more complex for repeated measures, so we'll use a simpler approach
        # Calculate total sum of squares
        grand_mean = repeated_measures_data['Score'].mean()
        ss_total = sum((repeated_measures_data['Score'] - grand_mean)**2)
        
        # Calculate between-subjects sum of squares
        ss_subjects = 0
        for subject in repeated_measures_data['Subject'].unique():
            subject_data = repeated_measures_data[repeated_measures_data['Subject'] == subject]['Score']
            subject_mean = subject_data.mean()
            ss_subjects += len(subject_data) * (subject_mean - grand_mean)**2
        
        # Calculate within-subjects sum of squares
        ss_within = ss_total - ss_subjects
        
        # Calculate time effect sum of squares
        ss_time = 0
        for time in repeated_measures_data['Time'].unique():
            time_data = repeated_measures_data[repeated_measures_data['Time'] == time]['Score']
            time_mean = time_data.mean()
            ss_time += len(time_data) * (time_mean - grand_mean)**2
        
        # Calculate error sum of squares
        ss_error = ss_within - ss_time

partial_eta_squared = ss_time / ss_error

print("\nEffect Size for Repeated Measures ANOVA:")
print(f"Partial Eta-squared: {partial_eta_squared:.4f}")

#%% Power analysis for ANOVA
# Calculate required sample size for one-way ANOVA
from statsmodels.stats.power import FTestPower

# Parameters
effect_size = eta_squared
alpha = 0.05
power = 0.8
n_groups = len(methods)

# For F-test power analysis, we need to specify df_num and df_denom
# For one-way ANOVA:
# df_num = k-1 (where k is the number of groups)
# df_denom = N-k (where N is the total sample size)
df_num = n_groups - 1

# We'll use a different approach since solve_power requires one parameter to be None
# Let's calculate power for a range of sample sizes and find the one that gives us our target power
n_range = np.arange(10, 100, 5)
power_values = []
analysis = FTestPower()

for n in n_range:
    total_n = n * n_groups
    df_denom = total_n - n_groups
    
    # For F-test power, we need to use the noncentrality parameter (ncp)
    # ncp = lambda * (df_num + 1) where lambda = effect_size * total_n
    ncp = effect_size * total_n * (df_num + 1)
    
    power_value = analysis.power(effect_size=effect_size, 
                                df_num=df_num,
                                df_denom=df_denom,
                                alpha=alpha,
                                ncc=ncp)  # Use ncc (noncentrality parameter) instead of nobs
    power_values.append(power_value)

# Find the required sample size (first n that gives power >= 0.8)
power_array = np.array(power_values)
required_n_idx = np.where(power_array >= 0.8)[0]
if len(required_n_idx) > 0:
    required_n = n_range[required_n_idx[0]]
else:
    required_n = n_range[-1]  # Use the largest n if we never reach 0.8 power

print("\nPower Analysis for One-Way ANOVA:")
print(f"Required sample size per group for 80% power: {required_n}")

# Visualize power curve
plt.figure(figsize=(10, 6))
plt.plot(n_range, power_values, 'b-', linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')
plt.axvline(x=required_n, color='g', linestyle='--', label=f'Required n={required_n}')
plt.title('Power Curve for One-Way ANOVA', fontsize=14)
plt.xlabel('Sample Size per Group')
plt.ylabel('Power')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('power_curve.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Conclusion
print("\nConclusion:")
print("1. One-Way ANOVA:")
if p_value < 0.05:
    print(f"   - There is a statistically significant difference between teaching methods (p = {p_value:.4f})")
    print(f"   - The effect size (eta-squared) is {eta_squared:.4f}, indicating a {('small' if eta_squared < 0.06 else 'medium' if eta_squared < 0.14 else 'large')} effect")
else:
    print(f"   - There is no statistically significant difference between teaching methods (p = {p_value:.4f})")

print("\n2. Two-Way ANOVA:")
# Check the actual column names in the ANOVA table
if 'PR(>F)' in anova_table.columns:
    method_p = anova_table.loc['Method', 'PR(>F)']
    study_time_p = anova_table.loc['StudyTime', 'PR(>F)']
    interaction_p = anova_table.loc['Method:StudyTime', 'PR(>F)']
else:
    # Try alternative column names
    if 'pvalue' in anova_table.columns:
        method_p = anova_table.loc['Method', 'pvalue']
        study_time_p = anova_table.loc['StudyTime', 'pvalue']
        interaction_p = anova_table.loc['Method:StudyTime', 'pvalue']
    elif 'p-value' in anova_table.columns:
        method_p = anova_table.loc['Method', 'p-value']
        study_time_p = anova_table.loc['StudyTime', 'p-value']
        interaction_p = anova_table.loc['Method:StudyTime', 'p-value']
    else:
        # If we can't find the right column, use a default value
        print("Warning: Could not find p-value column. Using default values.")
        method_p = 0.05
        study_time_p = 0.05
        interaction_p = 0.05

print(f"   - Main effect of teaching method: {'Significant' if method_p < 0.05 else 'Not significant'} (p = {method_p:.4f})")
print(f"   - Main effect of study time: {'Significant' if study_time_p < 0.05 else 'Not significant'} (p = {study_time_p:.4f})")
print(f"   - Interaction effect: {'Significant' if interaction_p < 0.05 else 'Not significant'} (p = {interaction_p:.4f})")

print("\n3. Repeated Measures ANOVA:")
# Check the actual column names in the repeated measures ANOVA table
if 'Pr > F' in aovrm.anova_table.columns:
    time_p = aovrm.anova_table.loc['Time', 'Pr > F']
else:
    # Try alternative column names
    if 'pvalue' in aovrm.anova_table.columns:
        time_p = aovrm.anova_table.loc['Time', 'pvalue']
    elif 'p-value' in aovrm.anova_table.columns:
        time_p = aovrm.anova_table.loc['Time', 'p-value']
    else:
        # If we can't find the right column, use a default value
        print("Warning: Could not find p-value column. Using default value.")
        time_p = 0.05

print(f"   - There is a {'statistically significant' if time_p < 0.05 else 'no statistically significant'} difference across time points (p = {time_p:.4f})")
print(f"   - The effect size (partial eta-squared) is {partial_eta_squared:.4f}, indicating a {('small' if partial_eta_squared < 0.06 else 'medium' if partial_eta_squared < 0.14 else 'large')} effect")

# %%
