#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%% Create sample data with multiple dimensions
dates = pd.date_range(start='2023-12-01', end='2023-12-31', freq='D')
n_days = len(dates)

# Generate sample data with multiple dimensions
data = {
    'date': dates,
    'gmv': [50000000 + i * 100000 + np.random.normal(0, 500000) for i in range(n_days)],
    'orders': [1000 + i * 10 + np.random.normal(0, 50) for i in range(n_days)],
    'users': [500 + i * 5 + np.random.normal(0, 20) for i in range(n_days)],
    'avg_order_value': [50000 + i * 100 + np.random.normal(0, 1000) for i in range(n_days)],
    'conversion_rate': [0.02 + i * 0.0001 + np.random.normal(0, 0.001) for i in range(n_days)],
    'customer_satisfaction': [4.5 + i * 0.01 + np.random.normal(0, 0.1) for i in range(n_days)],
    'marketing_spend': [1000000 + i * 5000 + np.random.normal(0, 50000) for i in range(n_days)],
    'category_a_sales': [20000000 + i * 50000 + np.random.normal(0, 200000) for i in range(n_days)],
    'category_b_sales': [15000000 + i * 40000 + np.random.normal(0, 150000) for i in range(n_days)],
    'category_c_sales': [15000000 + i * 10000 + np.random.normal(0, 100000) for i in range(n_days)]
}

df = pd.DataFrame(data)
df.set_index('date', inplace=True)
df.head()

#%% Display data preview and basic statistics
print("\nFirst few rows of the data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

#%% Create various visualizations
# 1. GMV Trend
plt.figure(figsize=(15, 8))
plt.bar(df.index, df['gmv'], color='navy', alpha=0.7)
plt.title('Daily GMV (December 2023)', fontsize=30)
plt.ylabel('GMV', fontsize=22)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
plt.xticks(rotation=0, fontsize=22)
plt.yticks(fontsize=18)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
plt.show()

# 2. Category Sales Comparison
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['category_a_sales'], label='Category A', linewidth=2)
plt.plot(df.index, df['category_b_sales'], label='Category B', linewidth=2)
plt.plot(df.index, df['category_c_sales'], label='Category C', linewidth=2)
plt.title('Category Sales Trends', fontsize=30)
plt.xlabel('Date', fontsize=22)
plt.ylabel('Sales', fontsize=22)
plt.legend(fontsize=18)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
plt.xticks(rotation=0, fontsize=22)
plt.yticks(fontsize=18)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
plt.show()

# 3. Key Metrics Correlation
plt.figure(figsize=(12, 8))
correlation_matrix = df[['gmv', 'orders', 'users', 'avg_order_value', 'conversion_rate']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Key Metrics Correlation', fontsize=20)
plt.tight_layout()
plt.show()

#%% Example Analysis
# Calculate daily growth rates
df['gmv_growth'] = df['gmv'].pct_change() * 100
df['orders_growth'] = df['orders'].pct_change() * 100

# Calculate ROI
df['roi'] = (df['gmv'] - df['marketing_spend']) / df['marketing_spend'] * 100

# Calculate category mix
df['category_a_mix'] = df['category_a_sales'] / df['gmv'] * 100
df['category_b_mix'] = df['category_b_sales'] / df['gmv'] * 100
df['category_c_mix'] = df['category_c_sales'] / df['gmv'] * 100

print("\nGrowth Rates and ROI:")
print(df[['gmv_growth', 'orders_growth', 'roi']].describe())
print("\nCategory Mix:")
print(df[['category_a_mix', 'category_b_mix', 'category_c_mix']].describe())



# %%
