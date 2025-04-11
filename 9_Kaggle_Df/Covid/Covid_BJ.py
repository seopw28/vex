
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Read and preprocess data
df = pd.read_csv('covid-19-all.csv')
df = df.fillna(0)  # Replace NaN values with 0

# Keep original date format for proper time series plotting
df['Date'] = pd.to_datetime(df['Date'])
df.head()


#%%
# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Confirmed Cases Over Time
axes[0,0].plot(df['Date'], df['Confirmed'], color='blue')
axes[0,0].set_title('Confirmed COVID-19 Cases in Beijing')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Number of Cases')
axes[0,0].tick_params(axis='x', rotation=45)

# Plot 2: Recovered Cases Over Time
axes[0,1].plot(df['Date'], df['Recovered'], color='green')
axes[0,1].set_title('Recovered COVID-19 Cases in Beijing')
axes[0,1].set_xlabel('Date')
axes[0,1].set_ylabel('Number of Cases')
axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: Deaths Over Time
axes[1,0].plot(df['Date'], df['Deaths'], color='red')
axes[1,0].set_title('COVID-19 Deaths in Beijing')
axes[1,0].set_xlabel('Date')
axes[1,0].set_ylabel('Number of Deaths')
axes[1,0].tick_params(axis='x', rotation=45)

# Plot 4: All metrics together
axes[1,1].plot(df['Date'], df['Confirmed'], label='Confirmed', color='blue')
axes[1,1].plot(df['Date'], df['Recovered'], label='Recovered', color='green')
axes[1,1].plot(df['Date'], df['Deaths'], label='Deaths', color='red')
axes[1,1].set_title('COVID-19 Overview in Beijing')
axes[1,1].set_xlabel('Date')
axes[1,1].set_ylabel('Number of Cases')
axes[1,1].legend()
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
# Save the figure as a PNG file with high DPI for better quality
plt.savefig('covid_beijing.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
