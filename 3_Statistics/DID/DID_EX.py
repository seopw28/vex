#%% Import libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

#%% Generate sample data
np.random.seed(42)

# Create time periods (0 for pre-treatment, 1 for post-treatment)
n_samples = 1000
time = np.random.randint(0, 2, n_samples)

# Create treatment group (0 for control, 1 for treatment)
treatment = np.random.randint(0, 2, n_samples)

# Create outcome variable with DID effect
baseline = 10
time_effect = 2
treatment_effect = 1
did_effect = 3

# Generate outcome with DID effect
outcome = (baseline + 
          time_effect * time + 
          treatment_effect * treatment + 
          did_effect * (time * treatment) + 
          np.random.normal(0, 1, n_samples))

# Create DataFrame
df = pd.DataFrame({
    'time': time,
    'treatment': treatment,
    'outcome': outcome
})

#%% Perform DID Analysis
# Create interaction term
df['time_x_treatment'] = df['time'] * df['treatment']

# Fit DID regression model
X = sm.add_constant(df[['time', 'treatment', 'time_x_treatment']])
model = sm.OLS(df['outcome'], X)
results = model.fit()

#%% Print results
print("\nDifference-in-Differences Analysis Results:")
print("==========================================")
print(results.summary().tables[1])

#%% Calculate and print group means
print("\nGroup Means:")
print("==========================================")
means = df.groupby(['time', 'treatment'])['outcome'].mean().unstack()
print("\nPre-treatment period (time=0):")
print(means.loc[0])
print("\nPost-treatment period (time=1):")
print(means.loc[1])

#%% Calculate DID manually for verification
control_diff = means.loc[1, 0] - means.loc[0, 0]
treatment_diff = means.loc[1, 1] - means.loc[0, 1]
did_estimate = treatment_diff - control_diff

print("\nManual DID Calculation:")
print("==========================================")
print(f"Control group difference (post - pre): {control_diff:.4f}")
print(f"Treatment group difference (post - pre): {treatment_diff:.4f}")
print(f"DID Estimate: {did_estimate:.4f}")

#%% Parallel trends assumption check (if you have pre-treatment data)
# Note: In real applications, you would need multiple pre-treatment periods
print("\nNote: Parallel trends assumption should be checked")
print("with multiple pre-treatment periods in real applications")

# %%
