#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style using seaborn's set_style instead of plt.style.use
sns.set_style("whitegrid")
sns.set_palette("husl")

#%% Create sample traffic performance data
# Generate date range for one month
dates = pd.date_range(start='2023-12-01', end='2023-12-31', freq='D')

# Generate sample data with realistic patterns
np.random.seed(42)  # For reproducibility

# Base values with daily variations
base_impressions = 1000000
base_clicks = 50000
base_unique_visitors = 80000
base_page_views = 1200000

# Create the dataset
data = {
    'date': dates,
    'impressions': [base_impressions + np.random.normal(0, 50000) for _ in range(len(dates))],
    'clicks': [base_clicks + np.random.normal(0, 2000) for _ in range(len(dates))],
    'unique_visitors': [base_unique_visitors + np.random.normal(0, 3000) for _ in range(len(dates))],
    'page_views': [base_page_views + np.random.normal(0, 60000) for _ in range(len(dates))]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate derived metrics
df['ctr'] = df['clicks'] / df['impressions'] * 100  # Click-through rate in percentage
df['pages_per_visit'] = df['page_views'] / df['unique_visitors']  # Average pages per visit

# Round numeric columns
numeric_columns = ['impressions', 'clicks', 'unique_visitors', 'page_views']
df[numeric_columns] = df[numeric_columns].round(0)
df[['ctr', 'pages_per_visit']] = df[['ctr', 'pages_per_visit']].round(2)

# Set date as index
df.set_index('date', inplace=True)
df.head()


#%% Create traffic performance visualization
# Create figure
plt.figure(figsize=(15, 8))

# Define colors
colors = sns.color_palette("husl", 4)

# Plot Traffic Volume Metrics
plt.plot(df.index, df['impressions'], label='Impressions', color=colors[0], linewidth=2)
plt.plot(df.index, df['unique_visitors'], label='Unique Visitors', color=colors[1], linewidth=2)
plt.plot(df.index, df['page_views'], label='Page Views', color=colors[2], linewidth=2)
plt.title('Daily Traffic Volume', pad=20, fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%% Save the plot
plt.savefig('traffic_performance.png', dpi=300, bbox_inches='tight')