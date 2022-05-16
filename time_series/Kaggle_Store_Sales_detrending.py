# SARIMAX - From here
import matplotlib.pyplot as plt
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

plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)

#Copy of dataset
Train = pd.read_csv('train.csv', parse_dates=["date"])
Train = Train.set_index("date").to_period("D")

stores = Train.store_nbr.unique()
families = Train.family.unique()

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
            models_SARIMAX.append(sm.tsa.statespace.SARIMAX(X.sales, order=(1,1,1), seasonal_order=(1, 0, 1, 7)))
            #models_SARIMAX.append(sm.tsa.statespace.SARIMAX(X.sales, order=(1,1,1)))
            model_trained += 1
            datasets.append(X.sales)

#Memory issues starts here:

for j in range(100):
    #@timing
    results.append(models_SARIMAX[j].fit(disp=False))
    #print(f'{j} / 1781')

results[2].plot_diagnostics(figsize=(16, 8))
plt.show()
