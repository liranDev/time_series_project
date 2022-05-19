# SARIMAX - From here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sm as sm
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL

#from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

from statsmodels.graphics.tsaplots import plot_acf
#import statsmodels.tsa.api as smt

register_matplotlib_converters()
sns.set_style("darkgrid")
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import multiprocessing

plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)

#Copy of dataset
Train = pd.read_csv('data/train.csv', parse_dates=["date"])
Train = Train.set_index("date").to_period("D")

stores = Train.store_nbr.unique()
families = Train.family.unique()

def getResultes(models):
    for j in range(LEN):
        results.append(models[j].fit(disp=False))
        print(f'{j} / 1781')

i = 0
datasets = []
models_SARIMAX = []
results = []
LEN = 5
model_trained = 0
for store in stores:
    is_store_nbr = Train['store_nbr'] == store
    Filtered_df = Train[is_store_nbr].copy()
    for family in families:
        single_test = Filtered_df.loc[Filtered_df['family'] == family]
        X = single_test.copy()
        X["dayofyear"] = X.index.dayofyear
        X["year"] = X.index.year
        #print(X.sales, family)
        if X.sales.mean() > 2:
            #models_SARIMAX.append(sm.tsa.statespace.SARIMAX(X.sales, trend="c", order=(1,1,1), seasonal_order=(1, 0, 1, 7)))
            models_SARIMAX.append(sm.tsa.statespace.SARIMAX(X.sales, trend="c", order=(1,1,1), seasonal_order=(1, 0, 1, 7)))
            #models_SARIMAX.append(sm.tsa.statespace.SARIMAX(X.sales, order=(1,1,1)))
            model_trained += 1
            datasets.append(X.sales)

cores = 2
j = 0
for i in range(cores):
    Processes = []
    p = multiprocessing.Process(target=getResultes, args=models_SARIMAX[j:round(i*len(models_SARIMAX)/cores)])
    j = j + len(models_SARIMAX)/cores
    Processes.append(p)
    p.start()



result = models_SARIMAX[239].fit(disp=False)
#results[238].plot_diagnostics(figsize=(16, 8))
result.plot_diagnostics(figsize=(16, 8))
plt.show()

k = 2
start_forecast = 100
pred = results[k].get_prediction(start=start_forecast, dynamic=False)
pred_ci = pred.conf_int()

ax = datasets[k].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Predictions', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()

plt.show()

#%% Forecast 16 days
days = 16
k=2
#current_dir = 'C:\Users\DULA\PycharmProjects\ex4-signals'
pred_uc = results[k].get_forecast(steps=1*days)
pred_ci = pred_uc.conf_int()
ax = datasets[k].plot(label='observed', figsize=(30, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()
#plt.savefig(current_dir + 'SARIMAX_FORECAST_1y.png')
plt.close()

print(results[k].summary())
print(pred_uc.predicted_mean)
