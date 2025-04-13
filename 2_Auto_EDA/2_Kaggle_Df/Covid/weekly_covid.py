
#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('covid.csv')
df.head()
# %%
# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
# Now convert to string format if needed
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

# Extract week number before converting to string
df['week_no'] = pd.to_datetime(df['Date']).dt.isocalendar().week
df['week_no'] = df['week_no'].astype(int)


df.head()

# %%
# Delete the 'Country/Region' and 'Province/State' columns
df_2 = df.drop(['Country/Region', 'Province/State'], axis=1)
df_2.head()

#%%
df_2 = df_2.fillna(0)
df_2.head()

# %%
# Create a copy of the dataframe to calculate daily values
daily_df = df_2.copy()

# Sort the dataframe by Date
daily_df = daily_df.sort_values(['Date'])

# Calculate daily values (difference between consecutive days)
# Since we already filled NaN values with 0 earlier, we just need to handle the first row
daily_df['Daily_Confirmed'] = daily_df['Confirmed'].diff()
daily_df.loc[daily_df.index[0], 'Daily_Confirmed'] = daily_df.loc[daily_df.index[0], 'Confirmed']

daily_df['Daily_Recovered'] = daily_df['Recovered'].diff()
daily_df.loc[daily_df.index[0], 'Daily_Recovered'] = daily_df.loc[daily_df.index[0], 'Recovered']

daily_df['Daily_Deaths'] = daily_df['Deaths'].diff()
daily_df.loc[daily_df.index[0], 'Daily_Deaths'] = daily_df.loc[daily_df.index[0], 'Deaths']

daily_df.head()


# %%
# Plot weekly values
plt.figure(figsize=(15, 10))

# Group data by week_no and sum the daily values
weekly_data = daily_df.groupby('week_no').agg({
    'Daily_Confirmed': 'sum',
    'Daily_Recovered': 'sum',
    'Daily_Deaths': 'sum'
}).reset_index()

# Plot weekly confirmed cases
plt.subplot(3, 1, 1)
plt.bar(weekly_data['week_no'], weekly_data['Daily_Confirmed'], color='blue')
plt.title('Weekly New Confirmed COVID-19 Cases')
plt.ylabel('New Cases')
plt.xlabel('Week Number')

# Plot weekly recovered cases
plt.subplot(3, 1, 2)
plt.bar(weekly_data['week_no'], weekly_data['Daily_Recovered'], color='green')
plt.title('Weekly New Recovered COVID-19 Cases')
plt.ylabel('New Recoveries')
plt.xlabel('Week Number')

# Plot weekly deaths
plt.subplot(3, 1, 3)
plt.bar(weekly_data['week_no'], weekly_data['Daily_Deaths'], color='red')
plt.title('Weekly New COVID-19 Deaths')
plt.ylabel('New Deaths')
plt.xlabel('Week Number')

plt.tight_layout()

plt.savefig('weekly_covid.png', dpi=300)

plt.show()

# %%
