#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   

# %%
# Load the data
df = pd.read_csv('wk_tx.csv')
df.head(10)

#%%

df_melted = pd.melt(df, 
                    id_vars=["Product_Code"],
                    var_name="Week", 
                    value_name="tx_count"
                    )

df_melted.tail()

# %%
# Apply condition to filter rows where Week column starts with 'W'
df2 = df_melted[df_melted['Week'].str[0] == 'W']
df2.tail()

# %%
# Remove 'P' from Product_Code values
df2['item_no'] = df2['Product_Code'].str.replace('P', '').astype(int)
df2.head()
# %%

df2['item_no'] = df2['item_no'].mod(5)
df2.tail()


# %%
# Create a line graph with Week on x-axis, tx_count on y-axis, and item_no as hue
plt.figure(figsize=(12, 6))
sns.lineplot(data=df2, x='Week'
                     , y='tx_count'
                     , hue='item_no'
                     , marker='o'
                     , palette='viridis'
                     , ci=None
                     )
plt.title('Transaction Count by Week and Item Number')
plt.ylabel('Transaction Count')
plt.xticks(rotation=45)
plt.legend(title='Item Number', fontsize=13)
plt.tight_layout()

plt.savefig('weekly_Trx.png', dpi=300)

plt.show()

# %%
