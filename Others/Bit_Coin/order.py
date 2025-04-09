
#%% Import libraries
import pyupbit
import pandas as pd
import pprint
pd.options.display.float_format = "{:.0f}".format

#%% Login
f = open("test_btc.csv")
lines = f.readlines()
access = lines[0].strip()
secret = lines[1].strip()
f.close()
upbit = pyupbit.Upbit(access, secret)

#%% Check Balance
upbit.get_balance("KRW") 



# %%
