#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

#%% Load and prepare data
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

# Display the first few rows of the dataset
print("Sample of the Iris dataset:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nFeature names:", feature_names)
print("Target names:", target_names)

#%% Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

#%% Train the decision tree
# Create and train the decision tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

#%% Evaluate the model
# Make predictions
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

#%% Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Iris Classification", fontsize=16)
plt.savefig('decision_tree_iris.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Feature importance visualization
# Get feature importances
importances = dt_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances in Decision Tree", fontsize=14)
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Decision boundary visualization (for 2 features)
# Select two features for visualization
feature1_idx = 0  # sepal length
feature2_idx = 1  # sepal width

# Create a mesh grid
x_min, x_max = X[:, feature1_idx].min() - 1, X[:, feature1_idx].max() + 1
y_min, y_max = X[:, feature2_idx].min() - 1, X[:, feature2_idx].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Create a classifier with only 2 features
dt_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_2d.fit(X_train[:, [feature1_idx, feature2_idx]], y_train)

# Predict for each point in the mesh grid
Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, feature1_idx], X[:, feature2_idx], c=y, alpha=0.8)
plt.xlabel(feature_names[feature1_idx])
plt.ylabel(feature_names[feature2_idx])
plt.title("Decision Boundary (using only 2 features)", fontsize=14)
plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Cross-validation to find optimal max_depth
from sklearn.model_selection import cross_val_score

# Test different max_depth values
max_depths = range(1, 10)
cv_scores = []

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(max_depths, cv_scores, 'o-')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy vs. Max Depth', fontsize=14)
plt.grid(True)
plt.savefig('cross_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# Find optimal max_depth
optimal_depth = max_depths[np.argmax(cv_scores)]
print(f"\nOptimal max_depth: {optimal_depth}")
print(f"Best cross-validation accuracy: {max(cv_scores):.4f}")

#%% Train final model with optimal parameters
final_dt = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
final_dt.fit(X_train, y_train)

# Evaluate final model
y_pred_final = final_dt.predict(X_test)
accuracy_final = accuracy_score(y_test, y_pred_final)
print(f"\nFinal Model Accuracy: {accuracy_final:.4f}")

# Visualize the final decision tree
plt.figure(figsize=(20, 10))
plot_tree(final_dt, 
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title(f"Final Decision Tree (max_depth={optimal_depth})", fontsize=16)
plt.savefig('final_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show() 
# %%
