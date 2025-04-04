#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Create sample ad performance data
# Generate date range for one month
dates = pd.date_range(start='2023-12-01', end='2023-12-31', freq='D')

# Generate sample data with realistic patterns
np.random.seed(42)  # For reproducibility

# Base values with daily variations
base_impressions = 1000000
base_clicks = 50000
base_conversions = 1000
base_revenue = 5000000

# Create the dataset
data = {
    'date': dates,
    'impression': [base_impressions + np.random.normal(0, 50000) for _ in range(len(dates))],
    'click': [base_clicks + np.random.normal(0, 2000) for _ in range(len(dates))],
    'conversion_cnt': [base_conversions + np.random.normal(0, 100) for _ in range(len(dates))],
    'rev_amount': [base_revenue + np.random.normal(0, 200000) for _ in range(len(dates))],
    'transaction_count': [base_conversions + np.random.normal(0, 50) for _ in range(len(dates))]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate derived metrics
df['ctr'] = df['click'] / df['impression'] * 100  # Click-through rate in percentage
df['conversion_rate'] = df['conversion_cnt'] / df['click'] * 100  # Conversion rate in percentage

# Round numeric columns
numeric_columns = ['impression', 'click', 'conversion_cnt', 'rev_amount', 'transaction_count']
df[numeric_columns] = df[numeric_columns].round(0)
df[['ctr', 'conversion_rate']] = df[['ctr', 'conversion_rate']].round(2)

# Set date as index
df.set_index('date', inplace=True)

#%% 1. Daily Performance Trends
plt.figure(figsize=(15, 10))

# Impression and Click trends
plt.subplot(2, 2, 1)
plt.plot(df.index, df['impression'], label='Impressions', color='blue')
plt.plot(df.index, df['click'], label='Clicks', color='red')
# plt.title('Daily Impressions and Clicks')
# plt.legend()
# plt.xticks(rotation=45)

# CTR and Conversion Rate trends
plt.subplot(2, 2, 2)
plt.plot(df.index, df['ctr'], label='CTR', color='green')
plt.plot(df.index, df['conversion_rate'], label='Conversion Rate', color='purple')
plt.title('Daily CTR and Conversion Rate')
plt.legend()
plt.xticks(rotation=45)

# Revenue trend
plt.subplot(2, 2, 3)
plt.plot(df.index, df['rev_amount'], label='Revenue', color='orange')
plt.title('Daily Revenue')
plt.legend()
plt.xticks(rotation=45)

# Transaction count trend
plt.subplot(2, 2, 4)
plt.plot(df.index, df['transaction_count'], label='Transactions', color='brown')
plt.title('Daily Transaction Count')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
