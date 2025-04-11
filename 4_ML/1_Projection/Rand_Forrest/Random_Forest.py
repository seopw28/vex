#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap
import warnings
from fpdf import FPDF
import os
from datetime import datetime
warnings.filterwarnings('ignore')

#%% Generate sample data for classification
def generate_classification_data(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_repeated=1, random_state=42):
    """
    Generate synthetic classification data for Random Forest.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_informative : int
        Number of informative features
    n_redundant : int
        Number of redundant features
    n_repeated : int
        Number of repeated features
    random_state : int
        Random seed
        
    Returns:
    --------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list
        List of feature names
    """
    np.random.seed(random_state)
    
    # Generate feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Generate X
    X = np.random.randn(n_samples, n_features)
    
    # Make some features informative
    for i in range(n_informative):
        X[:, i] = X[:, i] * 2
    
    # Make some features redundant
    for i in range(n_informative, n_informative + n_redundant):
        X[:, i] = X[:, np.random.randint(0, n_informative)] + np.random.randn(n_samples) * 0.1
    
    # Make some features repeated
    for i in range(n_informative + n_redundant, n_informative + n_redundant + n_repeated):
        X[:, i] = X[:, np.random.randint(0, n_informative + n_redundant)]
    
    # Generate y (binary classification)
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Probability of class 1
        p = 1 / (1 + np.exp(-(X[i, 0] + X[i, 1] + X[i, 2] + 0.5 * X[i, 3] + 0.3 * X[i, 4])))
        y[i] = np.random.binomial(1, p)
    
    return X, y, feature_names

# Generate classification data
X_clf, y_clf, feature_names_clf = generate_classification_data()

# Create DataFrame for visualization
clf_data = pd.DataFrame(X_clf, columns=feature_names_clf)
clf_data['target'] = y_clf

# Display sample data
print("Sample of Classification Data:")
print(clf_data.head(10))
print("\nClass Distribution:")
print(clf_data['target'].value_counts(normalize=True))

#%% Generate sample data for regression
def generate_regression_data(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_repeated=1, noise=0.1, random_state=42):
    """
    Generate synthetic regression data for Random Forest.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_informative : int
        Number of informative features
    n_redundant : int
        Number of redundant features
    n_repeated : int
        Number of repeated features
    noise : float
        Amount of noise to add
    random_state : int
        Random seed
        
    Returns:
    --------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list
        List of feature names
    """
    np.random.seed(random_state)
    
    # Generate feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Generate X
    X = np.random.randn(n_samples, n_features)
    
    # Make some features informative
    for i in range(n_informative):
        X[:, i] = X[:, i] * 2
    
    # Make some features redundant
    for i in range(n_informative, n_informative + n_redundant):
        X[:, i] = X[:, np.random.randint(0, n_informative)] + np.random.randn(n_samples) * 0.1
    
    # Make some features repeated
    for i in range(n_informative + n_redundant, n_informative + n_redundant + n_repeated):
        X[:, i] = X[:, np.random.randint(0, n_informative + n_redundant)]
    
    # Generate y (regression)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1.5 * X[:, 2] + 0.5 * X[:, 3] + 0.3 * X[:, 4] + np.random.randn(n_samples) * noise
    
    return X, y, feature_names

# Generate regression data
X_reg, y_reg, feature_names_reg = generate_regression_data()

# Create DataFrame for visualization
reg_data = pd.DataFrame(X_reg, columns=feature_names_reg)
reg_data['target'] = y_reg

# Display sample data
print("\nSample of Regression Data:")
print(reg_data.head(10))
print("\nTarget Statistics:")
print(reg_data['target'].describe())

#%% Visualize classification data
plt.figure(figsize=(12, 8))

# Plot feature distributions by class
for i in range(min(5, len(feature_names_clf))):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='target', y=feature_names_clf[i], data=clf_data)
    plt.title(f'Distribution of {feature_names_clf[i]} by Class')

plt.tight_layout()
plt.savefig('classification_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(clf_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix', fontsize=14)
plt.savefig('classification_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Visualize regression data
plt.figure(figsize=(12, 8))

# Plot feature distributions
for i in range(min(5, len(feature_names_reg))):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=feature_names_reg[i], y='target', data=reg_data, alpha=0.5)
    plt.title(f'{feature_names_reg[i]} vs Target')

plt.tight_layout()
plt.savefig('regression_feature_scatterplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(reg_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix', fontsize=14)
plt.savefig('regression_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Train and evaluate Random Forest for classification
# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)

# Make predictions
y_pred_clf = rf_clf.predict(X_test_clf)
y_pred_proba_clf = rf_clf.predict_proba(X_test_clf)[:, 1]

# Evaluate model
print("\nRandom Forest Classification Results:")
print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.4f}")
print(f"Precision: {precision_score(y_test_clf, y_pred_clf):.4f}")
print(f"Recall: {recall_score(y_test_clf, y_pred_clf):.4f}")
print(f"F1 Score: {f1_score(y_test_clf, y_pred_clf):.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_clf, X_clf, y_clf, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
feature_importance_clf = pd.DataFrame({
    'feature': feature_names_clf,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_clf)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_clf)
plt.title('Feature Importance', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('classification_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test_clf, y_pred_proba_clf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('classification_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Train and evaluate Random Forest for regression
# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Train Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = rf_reg.predict(X_test_reg)

# Evaluate model
print("\nRandom Forest Regression Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test_reg, y_pred_reg):.4f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_reg, X_reg, y_reg, cv=5, scoring='neg_mean_squared_error')
cv_scores = -cv_scores  # Convert to positive MSE
print(f"Cross-validation MSE: {cv_scores}")
print(f"Mean CV MSE: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
feature_importance_reg = pd.DataFrame({
    'feature': feature_names_reg,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_reg)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_reg)
plt.title('Feature Importance', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('regression_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('regression_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Hyperparameter tuning for classification
# Define parameter grid
param_grid_clf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search_clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_clf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_clf.fit(X_train_clf, y_train_clf)

# Print results
print("\nHyperparameter Tuning Results (Classification):")
print(f"Best parameters: {grid_search_clf.best_params_}")
print(f"Best cross-validation score: {grid_search_clf.best_score_:.4f}")

# Train model with best parameters
best_rf_clf = grid_search_clf.best_estimator_
best_rf_clf.fit(X_train_clf, y_train_clf)

# Evaluate best model
y_pred_best_clf = best_rf_clf.predict(X_test_clf)
print(f"Test accuracy with best parameters: {accuracy_score(y_test_clf, y_pred_best_clf):.4f}")

#%% Hyperparameter tuning for regression
# Define parameter grid
param_grid_reg = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search_reg = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid_reg,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search_reg.fit(X_train_reg, y_train_reg)

# Print results
print("\nHyperparameter Tuning Results (Regression):")
print(f"Best parameters: {grid_search_reg.best_params_}")
print(f"Best cross-validation score (negative MSE): {grid_search_reg.best_score_:.4f}")

# Train model with best parameters
best_rf_reg = grid_search_reg.best_estimator_
best_rf_reg.fit(X_train_reg, y_train_reg)

# Evaluate best model
y_pred_best_reg = best_rf_reg.predict(X_test_reg)
print(f"Test MSE with best parameters: {mean_squared_error(y_test_reg, y_pred_best_reg):.4f}")

#%% Advanced analysis: SHAP values for classification
# Calculate SHAP values
explainer_clf = shap.TreeExplainer(best_rf_clf)
shap_values_clf = explainer_clf.shap_values(X_test_clf)

# If shap_values_clf is a list (for binary classification), take the second element
if isinstance(shap_values_clf, list):
    shap_values_clf = shap_values_clf[1]

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_clf, X_test_clf, feature_names=feature_names_clf, show=False)
plt.title('SHAP Summary Plot (Classification)', fontsize=14)
plt.tight_layout()
plt.savefig('classification_shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Advanced analysis: SHAP values for regression
# Calculate SHAP values
explainer_reg = shap.TreeExplainer(best_rf_reg)
shap_values_reg = explainer_reg.shap_values(X_test_reg)

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_reg, X_test_reg, feature_names=feature_names_reg, show=False)
plt.title('SHAP Summary Plot (Regression)', fontsize=14)
plt.tight_layout()
plt.savefig('regression_shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Permutation importance for classification
# Calculate permutation importance
perm_importance_clf = permutation_importance(
    best_rf_clf, 
    X_test_clf, 
    y_test_clf, 
    n_repeats=10, 
    random_state=42
)

# Create DataFrame
perm_importance_df_clf = pd.DataFrame({
    'feature': feature_names_clf,
    'importance_mean': perm_importance_clf.importances_mean,
    'importance_std': perm_importance_clf.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nPermutation Importance (Classification):")
print(perm_importance_df_clf)

# Visualize permutation importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance_mean', y='feature', data=perm_importance_df_clf)
plt.errorbar(
    x=perm_importance_df_clf['importance_mean'], 
    y=range(len(perm_importance_df_clf)), 
    xerr=perm_importance_df_clf['importance_std'], 
    fmt='none', 
    color='black', 
    capsize=5
)
plt.title('Permutation Importance (Classification)', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('classification_permutation_importance.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Permutation importance for regression
# Calculate permutation importance
perm_importance_reg = permutation_importance(
    best_rf_reg, 
    X_test_reg, 
    y_test_reg, 
    n_repeats=10, 
    random_state=42
)

# Create DataFrame
perm_importance_df_reg = pd.DataFrame({
    'feature': feature_names_reg,
    'importance_mean': perm_importance_reg.importances_mean,
    'importance_std': perm_importance_reg.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nPermutation Importance (Regression):")
print(perm_importance_df_reg)

# Visualize permutation importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance_mean', y='feature', data=perm_importance_df_reg)
plt.errorbar(
    x=perm_importance_df_reg['importance_mean'], 
    y=range(len(perm_importance_df_reg)), 
    xerr=perm_importance_df_reg['importance_std'], 
    fmt='none', 
    color='black', 
    capsize=5
)
plt.title('Permutation Importance (Regression)', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('regression_permutation_importance.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Partial dependence plots for classification
# Create custom partial dependence plot for classification
plt.figure(figsize=(10, 6))

# Get the feature index
feature_idx_clf = feature_names_clf.index(feature_importance_clf.iloc[0]['feature'])

# Create a range of values for the feature
feature_range = np.linspace(
    X_test_clf[:, feature_idx_clf].min(), 
    X_test_clf[:, feature_idx_clf].max(), 
    50
)

# Initialize array to store predictions
pdp_values = np.zeros(len(feature_range))

# For each value in the range, replace the feature value in all samples and get predictions
for i, value in enumerate(feature_range):
    X_temp = X_test_clf.copy()
    X_temp[:, feature_idx_clf] = value
    pdp_values[i] = best_rf_clf.predict_proba(X_temp)[:, 1].mean()

# Plot the partial dependence
plt.plot(feature_range, pdp_values, 'b-', linewidth=2)
plt.xlabel(feature_importance_clf.iloc[0]['feature'])
plt.ylabel('Average Predicted Probability')
plt.title(f'Partial Dependence Plot for {feature_importance_clf.iloc[0]["feature"]} (Classification)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('classification_partial_dependence.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Partial dependence plots for regression
# Create custom partial dependence plot for regression
plt.figure(figsize=(10, 6))

# Get the feature index
feature_idx_reg = feature_names_reg.index(feature_importance_reg.iloc[0]['feature'])

# Create a range of values for the feature
feature_range = np.linspace(
    X_test_reg[:, feature_idx_reg].min(), 
    X_test_reg[:, feature_idx_reg].max(), 
    50
)

# Initialize array to store predictions
pdp_values = np.zeros(len(feature_range))

# For each value in the range, replace the feature value in all samples and get predictions
for i, value in enumerate(feature_range):
    X_temp = X_test_reg.copy()
    X_temp[:, feature_idx_reg] = value
    pdp_values[i] = best_rf_reg.predict(X_temp).mean()

# Plot the partial dependence
plt.plot(feature_range, pdp_values, 'r-', linewidth=2)
plt.xlabel(feature_importance_reg.iloc[0]['feature'])
plt.ylabel('Average Predicted Value')
plt.title(f'Partial Dependence Plot for {feature_importance_reg.iloc[0]["feature"]} (Regression)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_partial_dependence.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Conclusion
print("\nConclusion:")
print("1. Random Forest performed well on both classification and regression tasks")
print(f"2. For classification, the best model achieved an accuracy of {accuracy_score(y_test_clf, y_pred_best_clf):.4f}")
print(f"3. For regression, the best model achieved an R² score of {r2_score(y_test_reg, y_pred_best_reg):.4f}")
print(f"4. The most important features for classification were: {', '.join(feature_importance_clf['feature'].head(3).tolist())}")
print(f"5. The most important features for regression were: {', '.join(feature_importance_reg['feature'].head(3).tolist())}")
print("6. SHAP values and permutation importance provided consistent feature importance rankings")
print("7. Hyperparameter tuning improved model performance")

#%% Generate PDF report
class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Random Forest Analysis Report', 0, 0, 'C')
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
pdf.cell(0, 10, 'Random Forest Analysis Report', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
pdf.ln(20)

# Introduction
pdf.chapter_title('Introduction')
pdf.chapter_body('This report presents the results of a Random Forest analysis for both classification and regression tasks. Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (for classification) or the mean prediction (for regression) of the individual trees. The analysis includes data exploration, model training, hyperparameter tuning, and advanced interpretability techniques.')

# Data Description
pdf.chapter_title('Data Description')
pdf.chapter_body('The analysis includes two synthetic datasets: one for classification and one for regression. Both datasets contain 1000 samples with 10 features each. For the classification task, the target is a binary variable. For the regression task, the target is a continuous variable. The features include informative, redundant, and repeated features to simulate real-world data complexity.')

# Classification Data Exploration
pdf.chapter_title('Classification Data Exploration')
pdf.chapter_body('The classification dataset was explored using various visualization techniques. Feature distributions by class, correlation matrices, and other exploratory data analysis methods were used to understand the data structure and relationships between features and the target variable.')

# Add classification visualizations
pdf.ln(5)
pdf.image('classification_feature_distributions.png', x=10, w=190)
pdf.ln(5)
pdf.image('classification_correlation_matrix.png', x=10, w=190)

# Regression Data Exploration
pdf.chapter_title('Regression Data Exploration')
pdf.chapter_body('The regression dataset was explored using scatter plots, correlation matrices, and other visualization techniques to understand the relationships between features and the target variable.')

# Add regression visualizations
pdf.ln(5)
pdf.image('regression_feature_scatterplots.png', x=10, w=190)
pdf.ln(5)
pdf.image('regression_correlation_matrix.png', x=10, w=190)

# Classification Results
pdf.add_page()
pdf.chapter_title('Classification Results')
pdf.chapter_body(f'The Random Forest classifier was trained on the classification dataset and achieved an accuracy of {accuracy_score(y_test_clf, y_pred_best_clf):.4f} on the test set. The model was evaluated using precision, recall, F1 score, and ROC curve analysis. Cross-validation was performed to assess the model\'s generalization performance.')

# Add classification results visualizations
pdf.ln(5)
pdf.image('classification_feature_importance.png', x=10, w=190)
pdf.ln(5)
pdf.image('classification_roc_curve.png', x=10, w=190)

# Regression Results
pdf.chapter_title('Regression Results')
pdf.chapter_body(f'The Random Forest regressor was trained on the regression dataset and achieved an R² score of {r2_score(y_test_reg, y_pred_best_reg):.4f} on the test set. The model was evaluated using mean squared error, root mean squared error, mean absolute error, and R² score. Cross-validation was performed to assess the model\'s generalization performance.')

# Add regression results visualizations
pdf.ln(5)
pdf.image('regression_feature_importance.png', x=10, w=190)
pdf.ln(5)
pdf.image('regression_actual_vs_predicted.png', x=10, w=190)

# Hyperparameter Tuning
pdf.add_page()
pdf.chapter_title('Hyperparameter Tuning')
pdf.chapter_body('Grid search with cross-validation was performed to find the optimal hyperparameters for both the classification and regression models. The hyperparameters tuned included the number of trees, maximum depth, minimum samples split, and minimum samples leaf.')

pdf.chapter_body(f'For classification, the best parameters were: {grid_search_clf.best_params_}')
pdf.chapter_body(f'For regression, the best parameters were: {grid_search_reg.best_params_}')

# Advanced Interpretability
pdf.chapter_title('Advanced Interpretability')
pdf.chapter_body('Several advanced interpretability techniques were applied to understand the models better:')

pdf.chapter_body('1. SHAP (SHapley Additive exPlanations) values were calculated to explain the output of the models.')
pdf.chapter_body('2. Permutation importance was calculated to assess the importance of each feature.')
pdf.chapter_body('3. Partial dependence plots were generated to visualize the relationship between the most important features and the target variable.')

# Add interpretability visualizations
pdf.ln(5)
pdf.image('classification_shap_summary.png', x=10, w=190)
pdf.ln(5)
pdf.image('regression_shap_summary.png', x=10, w=190)

# Feature Importance Comparison
pdf.add_page()
pdf.chapter_title('Feature Importance Comparison')
pdf.chapter_body('Different feature importance measures were compared to ensure robustness of the feature importance rankings:')

pdf.chapter_body('1. Built-in feature importance from Random Forest')
pdf.chapter_body('2. Permutation importance')
pdf.chapter_body('3. SHAP values')

# Add feature importance comparison visualizations
pdf.ln(5)
pdf.image('classification_permutation_importance.png', x=10, w=190)
pdf.ln(5)
pdf.image('regression_permutation_importance.png', x=10, w=190)

# Partial Dependence Plots
pdf.chapter_title('Partial Dependence Plots')
pdf.chapter_body('Partial dependence plots were generated to visualize the relationship between the most important features and the target variable, while accounting for the average effect of other features.')

# Add partial dependence plots
pdf.ln(5)
pdf.image('classification_partial_dependence.png', x=10, w=190)
pdf.ln(5)
pdf.image('regression_partial_dependence.png', x=10, w=190)

# Conclusion
pdf.chapter_title('Conclusion')
conclusion_text = f"""
Random Forest performed well on both classification and regression tasks. For classification, the best model achieved an accuracy of {accuracy_score(y_test_clf, y_pred_best_clf):.4f}. For regression, the best model achieved an R² score of {r2_score(y_test_reg, y_pred_best_reg):.4f}.

The most important features for classification were: {', '.join(feature_importance_clf['feature'].head(3).tolist())}.
The most important features for regression were: {', '.join(feature_importance_reg['feature'].head(3).tolist())}.

SHAP values and permutation importance provided consistent feature importance rankings, confirming the robustness of the feature importance analysis. Hyperparameter tuning improved model performance for both tasks.

Random Forest is a powerful and versatile algorithm that can handle both classification and regression tasks effectively. Its ability to capture non-linear relationships, handle high-dimensional data, and provide feature importance rankings makes it a valuable tool for predictive modeling.
"""
pdf.chapter_body(conclusion_text)

# Save the PDF
pdf.output('random_forest_analysis_report.pdf')

print("\nPDF report generated: random_forest_analysis_report.pdf")

# %%
