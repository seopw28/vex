import pyupbit
import pandas as pd
import pprint
pd.options.display.float_format = "{:.0f}".format

# 로그인
f = open("upbit.txt")
lines = f.readlines()
access = lines[0].strip()
secret = lines[1].strip()
f.close()
upbit = pyupbit.Upbit(access, secret)

# 잔고 조회 
upbit.get_balance("KRW") 


