#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('pokemon_df.csv')
df.head(10)


# %%
# Filter for Pokemon with secondary typing
df_2 = df[df['Secondary Typing Flag'] != False]
df_2.head()


#%%
# Calculate mean stats by typing combination
stats_df = df_2.groupby(['Primary Typing', 'Secondary Typing']).agg({
    'Weight (hg)': lambda x: round(x.mean(), 2),
    'Height (dm)': lambda x: round(x.mean(), 2),
    'Attack': lambda x: round(x.mean(), 2),
    'Defense': lambda x: round(x.mean(), 2)
}).reset_index()

stats_df.head(10)

# %%
