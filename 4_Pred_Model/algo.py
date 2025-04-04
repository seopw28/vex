import pyupbit
import pandas as pd
import pprint
import matplotlib.pyplot as plt
pd.options.display.float_format = "{:.0f}".format

# Get 30 days of daily data
df = pyupbit.get_ohlcv(ticker="KRW-BTC", 
                       interval="day",  # daily data
                       count=30  # last 30 days
                       )

# Create bar graph with improved styling
plt.figure(figsize=(15, 8))
plt.bar(df.index, df['close'], color='skyblue', alpha=0.7)
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

df.to_csv("test_btc.csv")
print(df)



target = pyupbit.get_tickers(fiat="KRW")
prices = pyupbit.get_current_price(target)

for x, y in prices.items():
	print(x,y)
 


od_bk = pyupbit.get_orderbook("KRW-BTC")
pprint.pprint(od_bk)

od_1 = od_bk["orderbook_units"][0] 
od_ask_tot = od_bk["total_ask_size"] 
od_bid_tot = od_bk["total_bid_size"]

