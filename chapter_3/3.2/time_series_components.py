import statsmodels.tsa.seasonal as sts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv('C:/Users/sbhobbes/Downloads/us_retail_employment.csv', parse_dates = ['Month'], dtype = {'Title' : str, 'Employed' : np.float64})
data.set_index('Month', inplace = True)

# Decompose the time series
decomposition = sts.seasonal_decompose(data['Employed'])

# Extract the trend component
trend = decomposition.trend

# Extract the seasonality component
seasonality = decomposition.seasonal

# Extract the noise component
noise = decomposition.resid

seasonal_adjust = trend + noise

df = pd.DataFrame(data)
df['season_adjust'] = seasonal_adjust
df['trend'] = trend
sns.lineplot(data = df, x = 'Month', y = 'Employed', color = 'gray', label = 'Employed')
sns.lineplot(data = df, x = 'Month', y = 'season_adjust', color = '#007282', label = 'Seasonally Adjusted')
sns.lineplot(data = df, x = 'Month', y = 'trend', color = '#D55E00', label = 'Trend')

plt.ylabel('Persons (thousands)')
plt.title('Total employment in US retail')
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()
