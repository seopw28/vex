#%% Imports
import pyupbit
import pandas as pd
import pprint
import matplotlib.pyplot as plt
pd.options.display.float_format = "{:.0f}".format

#%% Data Collection
# Get 30 days of daily data
df = pyupbit.get_ohlcv(ticker="KRW-BTC", 
                       interval="day",  # daily data
                       count=30  # last 30 days
                       )

# Name the index (date column)
df.index.name = "Date"

#%% Data Visualization
# Create line graph with improved styling
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['close'], color='skyblue', linewidth=2, marker='o')
plt.title('Bitcoin Daily Closing Prices (Last 30 Days)', fontsize=16, pad=15)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (KRW)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Format y-axis with comma separator for better readability
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.show()

#%% Data Export
df.to_csv("test_btc.csv")
print(df)


# %%

# %%
