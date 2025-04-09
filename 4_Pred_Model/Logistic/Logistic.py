#%% Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#%% Generate sample data
# Generate sample e-commerce data
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.normal(35, 10, n_samples)  # Age
time_spent = np.random.exponential(30, n_samples)  # Time spent on site
pages_visited = np.random.poisson(5, n_samples)  # Pages visited
cart_value = np.random.gamma(shape=2, scale=20, size=n_samples)  # Cart value

# Create some correlations between features and outcome
purchase_prob = 1 / (1 + np.exp(-(
    -2 +
    0.03 * age +
    0.02 * time_spent +
    0.2 * pages_visited +
    0.01 * cart_value
)))

# Generate binary outcome (purchase: 1, no purchase: 0)
purchase = np.random.binomial(1, purchase_prob)

#%% Create DataFrame and split data
# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'time_spent': time_spent,
    'pages_visited': pages_visited,
    'cart_value': cart_value,
    'purchase': purchase
})

# Split features and target
X = data.drop('purchase', axis=1)
y = data['purchase']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Train model and evaluate
# Create and train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%% Analyze feature importance
# Print feature coefficients
print("\nFeature Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Coefficient', ascending=True)

sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance in Logistic Regression Model')
plt.tight_layout()
plt.savefig('feature_importance.png')  # Save the plot as a PNG file
plt.show()

#%% Make predictions
# Example prediction for a new customer
new_customer = np.array([[30, 25, 4, 50]])  # age, time_spent, pages_visited, cart_value
prediction = model.predict_proba(new_customer)
print("\nPrediction for new customer:")
print(f"Probability of purchase: {prediction[0][1]:.2%}")

# %%
