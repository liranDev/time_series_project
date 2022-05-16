# SARIMAX - From here
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from pathlib import Path
import time
from statsmodels.tsa.seasonal import STL

# from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.tsa.api as smt

register_matplotlib_converters()
sns.set_style("darkgrid")
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

DATA_FILE = Path.cwd().parent / 'data' / 'train.csv'

train_df = pd.read_csv(DATA_FILE, parse_dates=["date"])
train_df = train_df.set_index("date").to_period("D")

stores = train_df.store_nbr.unique()
families = train_df.family.unique()

models_SARIMAX = []
results = []

LEN = 5

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))

        return ret

    return wrap


@timing
def train_model(index):
    return models_SARIMAX[index].fit(disp=False)


model_trained = 0
# store_bsr_as_df =
total = len(stores) * len(families)
for store in stores:
    is_store_nbr = train_df['store_nbr'] == store
    Filtered_df = train_df[is_store_nbr].copy()
    for family in families:
        single_test = Filtered_df.loc[Filtered_df['family'] == family]
        X = single_test.copy()
        X["dayofyear"] = X.index.dayofyear
        X["year"] = X.index.year
        # print(X.sales, family)
        models_SARIMAX.append(sm.tsa.statespace.SARIMAX(X.sales, trend="c", order=(1, 1, 1), seasonal_order=(1, 0, 1, 12),
                                     enforce_stationarity=False, enforce_invertibility=False))
        model_trained += 1

        if model_trained % LEN == 0:
            print(f'{model_trained} / {total}')
            break
    else:
        continue
    break

# Memory issues starts here:

# 1781
for j in range(LEN):
    results.append(train_model(j))
    print(f'{j} / 1781')
