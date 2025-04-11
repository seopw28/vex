#%%
import pandas as pd
import os

df = pd.read_csv("chocolate.csv")
df.head()

# %%


# Create pivot table of salesperson vs country with sum of boxes shipped
pivot_table = pd.pivot_table(df, 
                           values='Boxes Shipped',
                           index='Sales Person', 
                           columns='Country',
                           aggfunc='sum')
pivot_table.head()

#%%
# Import seaborn for heatmap visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Create heatmap with purple color scheme
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table,
            annot=True,  # Show values in cells 
            fmt='.0f',   # Format as integers
            cmap='Purples',  # Purple color scheme
            cbar_kws={'label': 'Boxes Shipped'})

plt.title('Boxes Shipped by Sales Person and Country')
plt.tight_layout()

# Save the plot as an image file
plt.savefig('sales_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
