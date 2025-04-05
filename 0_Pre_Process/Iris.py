#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

#%% Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()


#%% 1. Pair Plot Visualization
plt.figure(figsize=(12, 6))
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle('Iris Dataset - Pair Plot', y=1.02, fontsize=20)
plt.show()

#%% 2. Box Plots Visualization
plt.figure(figsize=(12, 6))
df.boxplot(by='species', figsize=(12, 6))
plt.title('Distribution of Features by Species')
plt.suptitle('')  # Remove automatic suptitle
plt.show()

#%% 3. Correlation Heatmap Visualization
plt.figure(figsize=(10, 8))
correlation = df.drop('species', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

#%% 4. Violin Plots Visualization
plt.figure(figsize=(12, 6))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='species', y=feature, data=df)
    plt.title(f'{feature} by Species')
plt.tight_layout()
plt.show()

#%% Statistical Analysis
# Calculate mean and standard deviation by species
print("\nMean values by species:")
print(df.groupby('species').mean())
print("\nStandard deviation by species:")
print(df.groupby('species').std())

# Calculate correlation with species
species_corr = pd.get_dummies(df['species']).corrwith(df.drop('species', axis=1))
print("\nCorrelation with species:")
print(species_corr)

#%% 5. Ratio Analysis and Visualization
# Calculate feature ratios
df['sepal_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']

# Plot ratios
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='species', y='sepal_ratio', data=df)
plt.title('Sepal Ratio by Species')

plt.subplot(1, 2, 2)
sns.boxplot(x='species', y='petal_ratio', data=df)
plt.title('Petal Ratio by Species')
plt.tight_layout()
plt.show()

# Print ratio statistics
print("\nRatio Statistics:")
print(df[['sepal_ratio', 'petal_ratio']].describe())

#%% Statistical Analysis
# Calculate mean and standard deviation by species
print("\nMean values by species:")
print(df.groupby('species').mean())
print("\nStandard deviation by species:")
print(df.groupby('species').std())

# Calculate correlation with species
species_corr = pd.get_dummies(df['species']).corrwith(df.drop('species', axis=1))
print("\nCorrelation with species:")
print(species_corr)

#%% Additional Analysis
# Calculate feature ratios
df['sepal_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']

# Plot ratios
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='species', y='sepal_ratio', data=df)
plt.title('Sepal Ratio by Species')

plt.subplot(1, 2, 2)
sns.boxplot(x='species', y='petal_ratio', data=df)
plt.title('Petal Ratio by Species')
plt.tight_layout()
plt.show()

# Print ratio statistics
print("\nRatio Statistics:")
print(df[['sepal_ratio', 'petal_ratio']].describe())