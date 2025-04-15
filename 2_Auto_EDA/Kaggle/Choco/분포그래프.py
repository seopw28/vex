#%%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('chocolate.csv')
df.head()

# %%
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df.head()

#%%
df['Amount'] = df['Amount'].str.replace('$', '')
df['Amount'] = df['Amount'].str.replace(',', '')
df['Amount'] = df['Amount'].astype(float)
df.head()

# %%
gr_df = df.groupby(['Date','Country'])['Amount'].sum().round(0).astype(int).reset_index()
gr_df.head()

# %%
# Create a histogram to visualize the distribution of 'Amount'
plt.figure(figsize=(10, 6))
plt.hist(gr_df['Amount'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Amount by Date and Country')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

# Add a vertical line for the mean
plt.axvline(gr_df['Amount'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {gr_df["Amount"].mean():.2f}')

# Add a vertical line for the median
plt.axvline(gr_df['Amount'].median(), color='green', linestyle='dashed', linewidth=1, label=f'Median: {gr_df["Amount"].median():.2f}')

plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('분포그래프.png', dpi=300, bbox_inches='tight')
plt.show()

# Display basic statistics of the Amount column
print("Amount Statistics:")
print(gr_df['Amount'].describe())


# %%
