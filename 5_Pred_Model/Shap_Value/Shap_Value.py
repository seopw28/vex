#%% import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

np.random.seed(42) # Set random seed for reproducibility

#%% Function to generate sample data for SHAP analysis
def generate_sample_data(n_samples=1000, n_features=10):
    """
    Generate sample data for SHAP analysis.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features to generate
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature data
    y : pandas.Series
        Target variable
    """
    # Generate feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Generate random feature data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Generate target variable with some non-linear relationships
    y = (
        0.3 * X['feature_0']**2 +  # Quadratic relationship
        0.5 * X['feature_1'] +     # Linear relationship
        0.2 * X['feature_2'] * X['feature_3'] +  # Interaction
        0.1 * np.sin(X['feature_4']) +  # Non-linear relationship
        0.05 * np.random.randn(n_samples)  # Noise
    )
    
    return X, y

#%% Generate sample data
print("Generating sample data...")
X, y = generate_sample_data(n_samples=1000, n_features=10)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

#%% Train a random forest model
print("Training random forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"R² score on training data: {train_score:.4f}")
print(f"R² score on test data: {test_score:.4f}")

#%% Calculate SHAP values
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

#%% Create a summary plot of SHAP values
print("Creating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar Plot)")
plt.tight_layout()
plt.savefig('shap_summary_bar.png')
print("SHAP summary bar plot saved to 'shap_summary_bar.png'")

#%% Create a detailed SHAP summary plot
print("Creating detailed SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance (Detailed)")
plt.tight_layout()
plt.savefig('shap_summary_detailed.png')
print("SHAP summary detailed plot saved to 'shap_summary_detailed.png'")

#%% Create a dependence plot for the most important feature
print("Creating SHAP dependence plot...")
# Find the most important feature
feature_importance = np.abs(shap_values).mean(0)
most_important_feature_idx = np.argmax(feature_importance)
most_important_feature = X_test.columns[most_important_feature_idx]

plt.figure(figsize=(10, 6))
shap.dependence_plot(
    most_important_feature_idx, 
    shap_values, 
    X_test,
    interaction_index=None,
    show=False
)
plt.title(f"SHAP Dependence Plot for {most_important_feature}")
plt.tight_layout()
plt.savefig('shap_dependence_plot.png')
print(f"SHAP dependence plot saved to 'shap_dependence_plot.png'")

#%% Create a force plot for a specific instance
print("Creating SHAP force plot...")
# Select a random instance
instance_idx = np.random.randint(0, len(X_test))
instance = X_test.iloc[instance_idx]
instance_shap_values = explainer.shap_values(instance)

# Create a force plot
plt.figure(figsize=(12, 4))
shap.force_plot(
    explainer.expected_value,
    instance_shap_values,
    instance,
    matplotlib=True,
    show=False
)
plt.title(f"SHAP Force Plot for Instance {instance_idx}")
plt.tight_layout()
plt.savefig('shap_force_plot.png')
print(f"SHAP force plot saved to 'shap_force_plot.png'")

#%% Create a decision plot for multiple instances
print("Creating SHAP decision plot...")
# Select a few random instances
n_instances = 5
instance_indices = np.random.choice(len(X_test), n_instances, replace=False)
instances = X_test.iloc[instance_indices]
instance_shap_values = explainer.shap_values(instances)

plt.figure(figsize=(12, 8))
shap.decision_plot(
    explainer.expected_value,
    instance_shap_values,
    instances,
    feature_names=list(X_test.columns),
    show=False
)
plt.title("SHAP Decision Plot for Multiple Instances")
plt.tight_layout()
plt.savefig('shap_decision_plot.png')
print("SHAP decision plot saved to 'shap_decision_plot.png'")

#%% Save SHAP values to CSV for further analysis
print("Saving SHAP values to CSV...")
shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
shap_df.to_csv('shap_values.csv', index=False)
print("SHAP values saved to 'shap_values.csv'")

#%% Function to analyze feature interactions
    
def analyze_feature_interactions(shap_values, X, top_n=3):
    print("Analyzing feature interactions...")


    """
    Analyze feature interactions using SHAP values.
    
    Parameters:
    -----------
    shap_values : numpy.ndarray
        SHAP values for the dataset
    X : pandas.DataFrame
        Feature data
    top_n : int
        Number of top interactions to return
        
    Returns:
    --------
    interactions : list
        List of top feature interactions
    """

    # 가장 중요한 feature 인덱스 선택
    feature_importance = np.abs(shap_values).mean(0)
    top_index = np.argmax(feature_importance)

    # 상호작용 계산
    interaction_values = shap.approximate_interactions(
        top_index,
        shap_values,
        X
    )

    feature_names = X.columns
    top_interactions = sorted(
        zip(feature_names[interaction_values], interaction_values),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top_interactions


    # Get feature names
    feature_names = X.columns
    
    # Calculate interaction strengths
    interaction_strengths = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            strength = np.abs(interaction_values[:, i, j]).mean()
            interaction_strengths.append({
                'feature1': feature_names[i],
                'feature2': feature_names[j],
                'strength': strength
            })
    
    # Sort by strength
    interaction_strengths.sort(key=lambda x: x['strength'], reverse=True)
    
    return interaction_strengths[:top_n]

#%% Analyze feature interactions
print("Analyzing feature interactions...")
top_interactions = analyze_feature_interactions(shap_values, X_test, top_n=3)

print("\nTop Feature Interactions:")
for i, interaction in enumerate(top_interactions):
    print(f"{i+1}. {interaction['feature1']} × {interaction['feature2']}: {interaction['strength']:.4f}")

#%% Main function to run the analysis
def main():
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=1000, n_features=10)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a random forest model
    print("Training random forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Create visualizations
    print("Creating SHAP visualizations...")
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png')
    
    # Detailed summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance (Detailed)")
    plt.tight_layout()
    plt.savefig('shap_summary_detailed.png')
    
    # Dependence plot
    feature_importance = np.abs(shap_values).mean(0)
    most_important_feature_idx = np.argmax(feature_importance)
    most_important_feature = X_test.columns[most_important_feature_idx]
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        most_important_feature_idx, 
        shap_values, 
        X_test,
        interaction_index=None,
        show=False
    )
    plt.title(f"SHAP Dependence Plot for {most_important_feature}")
    plt.tight_layout()
    plt.savefig('shap_dependence_plot.png')
    
    # Save SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df.to_csv('shap_values.csv', index=False)
    
    print("SHAP analysis completed successfully!")

if __name__ == "__main__":
    main() 