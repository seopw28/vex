#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('Daily_Attendance.csv')
df.head()

#%%
df_2 = df.groupby(['School DBN']).agg({'Enrolled':'sum',
                                       'Present':'sum'
                                       }).reset_index()
df_2.head()

#%%
df_2['at_rate'] = round(df_2['Present'] * 100 / df_2['Enrolled'], 1)
df_2.head()

#%%
df_3 = df_2[df_2['Enrolled'] > 1]
df_3.head()


# %%
# 두 개 기준으로 정렬: at_rate 내림차순 → Enrolled 내림차순
df_3 = df_3.sort_values(by=['at_rate', 'Enrolled'], ascending=[False, False])

# rank: 같은 at_rate일 경우 Enrolled 많은 순으로 반영되게 함
df_3['rank'] = df_3.reset_index(drop=True).index + 1
df_3.head(10)


#%%
df_3 = df_3[df_3['rank'] <= 10]
df_3.head(11)

# %%
# Import squarify for treemap visualization
import squarify

# Create a simple treemap visualization
plt.figure(figsize=(10, 6))

# Use enrollment numbers for box sizes
sizes = df_3['Enrolled']

# Create simple labels with school ID and attendance rate
labels = [f"{dbn}: {rate}%" 
          for dbn, rate in zip(df_3['School DBN'], df_3['at_rate'])]

# Use a basic color scheme
colors = plt.cm.Paired(range(len(df_3)))

# Plot the treemap with simplified parameters and add padding between squares
squarify.plot(sizes=sizes, 
              label=labels, 
              color=colors, 
              alpha=0.7,
              pad=0.05)  # Add padding between squares

# Add a simple title and remove axes
plt.title('Top 10 Schools by Attendance Rate')
plt.axis('off')

# Save and display the figure
plt.tight_layout()
plt.savefig('top_schools_treemap.png')
plt.show()
# %%
