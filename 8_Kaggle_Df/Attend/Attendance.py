#%%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('Daily_Attendance.csv')

# %%
df['at_rate'] = round(df['Present'] * 100 / df['Enrolled'], 1)
df.head()

# %%
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df.head()

#%%
# Create 'year_week' column
df['year_week'] = df['Date'].dt.strftime('%Y-W%U')

# Group by 'year_week' and aggregate
df_agg = df.groupby(['year_week']).agg(
    {'Enrolled': 'sum', 
     'Present': 'sum', 
     'Absent': 'sum'
     }
    ).reset_index()

# Remove rows with specific year_week
df_agg = df_agg[~df_agg['year_week'].isin(['2019-W06', '2019-W07'])]
df_agg.head()


#%%
# Calculate attendance rate
df_agg['at_rate'] = round(df_agg['Present'] * 100 / df_agg['Enrolled'], 1)

# Plot the attendance rate by year_week
plt.figure(figsize=(12, 8))
plt.plot(df_agg['year_week'], df_agg['at_rate'], marker='o', linestyle='-')
plt.xlabel('Year-Week')
plt.ylabel('Attendance Rate (%)')
plt.title('Weekly Attendance Rate')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the plot as an image file
plt.savefig('weekly_att_rate.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
