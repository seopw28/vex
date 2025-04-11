#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Create a sample dataframe for demonstration
print("Creating a sample dataframe...")
data = {
    'Product': ['A', 'B', 'C', 'D'],
    'Q1_Sales': [100, 150, 120, 90],
    'Q2_Sales': [110, 140, 115, 95],
    'Q3_Sales': [105, 160, 125, 100],
    'Q4_Sales': [120, 170, 130, 110]
}
df = pd.DataFrame(data)
df.head()

#%%
# Using melt to convert from wide to long format
df_melted = pd.melt(
    df,
    id_vars=['Product'],  # Column(s) to use as identifier variables
    value_vars=['Q1_Sales', 'Q2_Sales', 'Q3_Sales', 'Q4_Sales'],  # Columns to unpivot
    var_name='Quarter',  # Name for the variable column
    value_name='Sales'  # Name for the value column
)
df_melted.head()

#%%
# Optional: Clean up the Quarter column by removing '_Sales'
df_melted['Quarter'] = df_melted['Quarter'].str.replace('_Sales', '')
df_melted.head()


#%%
# Visualize the melted data
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Product', y='Sales', hue='Quarter')
plt.title('Sales by Product and Quarter')
plt.tight_layout()
plt.savefig('melt_example.png', dpi=300)
print("Saved visualization as 'melt_example.png'")
print("\n")


#%%
# Create a long-format dataframe for pivot demonstration
data_long = {
    'Date': ['2023-01-01', '2023-01-01', '2023-01-01', 
             '2023-02-01', '2023-02-01', '2023-02-01',
             '2023-03-01', '2023-03-01', '2023-03-01'],
    'Category': ['Electronics', 'Clothing', 'Food',
                'Electronics', 'Clothing', 'Food',
                'Electronics', 'Clothing', 'Food'],
    'Sales': [1000, 800, 500, 1200, 900, 600, 1100, 950, 550]
}
df_long = pd.DataFrame(data_long)
df_long.head()

#%%
# Using pivot to convert from long to wide format
df_pivot = df_long.pivot(
    index='Date',  # Values from this column become the new index
    columns='Category',  # Values from this column become the new columns
    values='Sales'  # Values from this column fill the new table
)

df_pivot.head()

#%%
# Reset index to make Date a column again
df_pivot.reset_index(inplace=True)
df_pivot.head()

#%%
# Visualize the pivot data
plt.figure(figsize=(10, 6))
df_pivot.set_index('Date').plot(kind='bar')
plt.title('Sales by Category Over Time')
plt.ylabel('Sales')
plt.tight_layout()
plt.savefig('pivot_example.png', dpi=300)
print("Saved visualization as 'pivot_example.png'")

#%%
# ---- PIVOT_TABLE EXAMPLE ----
print("\n")
print("=" * 50)
print("PIVOT_TABLE EXAMPLE: With aggregation functions")
print("=" * 50)

#%%
# Create a dataframe with duplicate entries for pivot_table demonstration
data_dup = {
    'Date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01',
             '2023-02-01', '2023-02-01', '2023-02-01', '2023-02-01',
             '2023-03-01', '2023-03-01', '2023-03-01', '2023-03-01'],
    'Category': ['Electronics', 'Electronics', 'Clothing', 'Food',
                'Electronics', 'Clothing', 'Clothing', 'Food',
                'Electronics', 'Clothing', 'Food', 'Food'],
    'Sales': [1000, 1200, 800, 500, 1200, 900, 950, 600, 1100, 950, 550, 600]
}
df_dup = pd.DataFrame(data_dup)
print("Dataframe with duplicate entries:")
print(df_dup)
print("\n")

#%%
# Using pivot_table to handle duplicates with aggregation
df_pivot_table = pd.pivot_table(
    df_dup,
    index='Date',
    columns='Category',
    values='Sales',
    aggfunc='mean'  # Can use 'sum', 'count', 'min', 'max', etc.
)

df_pivot_table.head()

#%%
# Example with multiple aggregation functions
df_multi_agg = pd.pivot_table(
    df_dup,
    index='Date',
    columns='Category',
    values='Sales',
    aggfunc=[np.mean, np.sum, np.max]
)

df_multi_agg.head()

# %%
