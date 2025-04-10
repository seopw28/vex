#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Generate sample data
np.random.seed(42)
date_rng = pd.date_range(start='1/1/2020', end='1/01/2022', freq='D')
data = np.random.randn(len(date_rng)) + np.linspace(0, 10, len(date_rng))
df = pd.DataFrame(date_rng, columns=['date'])
df['data'] = data
df.set_index('date', inplace=True)
df.head()

#%%
# Fit ARIMA model
model = ARIMA(df['data'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, inclusive='right')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['data'], label='Original Data')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig('arima_forecast.png')
plt.show()

# %%
