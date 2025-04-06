#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

#%% Generate sample data
# Let's create a dataset with multiple features and a target variable
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate features
X1 = np.random.normal(50, 10, n_samples)  # Feature 1: Age
X2 = np.random.normal(30, 5, n_samples)   # Feature 2: Years of Experience
X3 = np.random.normal(15, 3, n_samples)   # Feature 3: Education Level
X4 = np.random.normal(8, 2, n_samples)    # Feature 4: Hours Worked per Week

# Generate target variable (Salary) with some noise
# y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₄X₄ + ε
beta0 = 20000  # Intercept
beta1 = 500    # Coefficient for Age
beta2 = 1000   # Coefficient for Years of Experience
beta3 = 2000   # Coefficient for Education Level
beta4 = 100    # Coefficient for Hours Worked
noise = np.random.normal(0, 5000, n_samples)  # Random noise

y = beta0 + beta1*X1 + beta2*X2 + beta3*X3 + beta4*X4 + noise

# Create a DataFrame
data = pd.DataFrame({
    'Age': X1,
    'Experience': X2,
    'Education': X3,
    'Hours_Worked': X4,
    'Salary': y
})

# Display sample data
print("Sample of the Dataset:")
print(data.head(10))
print("\nData Summary:")
print(data.describe())

#%% Data visualization
# Pair plot to see relationships between variables
plt.figure(figsize=(12, 8))
sns.pairplot(data, diag_kind='kde')
plt.suptitle('Pair Plot of Features and Target', y=1.02, fontsize=16)
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap', fontsize=16)
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Prepare data for modeling
# Separate features and target
X = data.drop('Salary', axis=1)
y = data['Salary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier interpretation
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\nTraining Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

#%% Train the model
# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Print model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

#%% Evaluate the model
# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-validation R-squared scores: {cv_scores}")
print(f"Average CV R-squared: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

#%% Visualize model results
# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary', fontsize=16)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residuals Plot', fontsize=16)
plt.tight_layout()
plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance', fontsize=16)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Detailed statistical analysis with statsmodels
# For statsmodels, we'll use numpy arrays to avoid index alignment issues
# Add constant for intercept
X_train_with_const = sm.add_constant(X_train_scaled.values)
X_test_with_const = sm.add_constant(X_test_scaled.values)

# Convert y_train to numpy array
y_train_array = y_train.values

# Fit the model
stats_model = sm.OLS(y_train_array, X_train_with_const).fit()

# Print summary
print("\nDetailed Statistical Analysis:")
print(stats_model.summary())

#%% Confidence intervals for coefficients
# Get confidence intervals
conf_int = stats_model.conf_int(alpha=0.05)
conf_int_df = pd.DataFrame(conf_int, columns=['Lower CI', 'Upper CI'])
conf_int_df.index = ['Intercept'] + list(X.columns)

print("\n95% Confidence Intervals for Coefficients:")
print(conf_int_df)

# Visualize confidence intervals
plt.figure(figsize=(12, 6))
coef_ci = pd.DataFrame({
    'Coefficient': model.coef_,
    'Lower CI': conf_int_df.iloc[1:, 0],
    'Upper CI': conf_int_df.iloc[1:, 1]
})
coef_ci.index = X.columns

# Plot coefficients with confidence intervals
plt.errorbar(x=coef_ci.index, y=coef_ci['Coefficient'], 
             yerr=[coef_ci['Coefficient'] - coef_ci['Lower CI'], 
                   coef_ci['Upper CI'] - coef_ci['Coefficient']],
             fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Coefficient Estimates with 95% Confidence Intervals', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('coefficient_ci.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Predictions for new data
# Create a sample new data point
new_data = pd.DataFrame({
    'Age': [45],
    'Experience': [15],
    'Education': [18],
    'Hours_Worked': [40]
})

# Scale the new data
new_data_scaled = scaler.transform(new_data)
new_data_scaled = pd.DataFrame(new_data_scaled, columns=new_data.columns)

# Make prediction
predicted_salary = model.predict(new_data_scaled)[0]

print("\nPrediction for New Data:")
print(new_data)
print(f"Predicted Salary: ${predicted_salary:,.2f}")

#%% Compare with true coefficients
print("\nComparison with True Coefficients:")
comparison = pd.DataFrame({
    'Feature': ['Intercept'] + list(X.columns),
    'True Coefficient': [beta0, beta1, beta2, beta3, beta4],
    'Estimated Coefficient': [model.intercept_] + list(model.coef_)
})
print(comparison)

#%% Conclusion
print("\nConclusion:")
print("1. The multiple linear regression model has been successfully trained and evaluated.")
print("2. The model explains approximately {:.2f}% of the variance in salary.".format(r2 * 100))
print("3. The most important features for predicting salary are:")
for feature, importance in zip(feature_importance['Feature'], feature_importance['Importance']):
    print(f"   - {feature}: {importance:.2f}")
print("4. The model can be used to make predictions for new data points.")
print("5. The confidence intervals provide a range of plausible values for each coefficient.") 