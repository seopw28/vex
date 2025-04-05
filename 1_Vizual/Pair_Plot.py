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