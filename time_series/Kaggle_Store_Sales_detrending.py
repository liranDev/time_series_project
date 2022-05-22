# SARIMAX - From here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels as sm
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
from utils import timeit
import pickle

#from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

from statsmodels.graphics.tsaplots import plot_acf
#import statsmodels.tsa.api as smt

# register_matplotlib_converters()
# sns.set_style("darkgrid")

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import os
import multiprocessing
from pathlib import Path
DATA_FILE_train = Path.cwd().parent / 'data' / 'train.csv'
DATA_FILE_test = Path.cwd().parent / 'data' / 'test.csv'
MODELS = Path.cwd().parent / 'models'
CPU_COUNT = multiprocessing.cpu_count()

if not Path(MODELS).is_dir():
    os.mkdir(MODELS)

# plt.rc("figure", figsize=(16, 12))
# plt.rc("font", size=13)

#Copy of dataset
Train = pd.read_csv(DATA_FILE_train, parse_dates=["date"])
Train = Train.set_index("date").to_period("D")
stores = Train.store_nbr.unique()
families = Train.family.unique()
#dataset for finding ID's
Test = pd.read_csv(DATA_FILE_test, parse_dates=["date"])
Test = Test.set_index("date").to_period("D")


#def getResultes(models):
#    tempResults = []
#    for j in range(LEN):
#        tempResults.append(models[j].fit(disp=False))
#        print(f'{j} / 1781')
#    return tempResults

def predict(res,days):
    pred = res.get_prediction(start=days, dynamic=False)
    pred_ci = pred.conf_int()

def forecast(res,days):
    #days = 16
    k = 2
    pred_uc = res.get_forecast(steps=1 * days)
    pred_ci = pred_uc.conf_int()
    return pred_uc.predicted_mean[4:]

i = 0
datasets = []
models_SARIMAX = []
results = []
LEN = 5
model_trained = 0
ModelDetailsList = []
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
            models_SARIMAX.append(sm.tsa.statespace.SARIMAX(X.sales, trend="c", order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)))
            #models_SARIMAX.append(sm.tsa.statespace.SARIMAX(X.sales, order=(1,1,1)))
            model_trained += 1
            datasets.append(X.sales)
            ModelDetailsList.append((store, family))

cores = 2
j = 0
first = models_SARIMAX[0:20]
second = models_SARIMAX[101:200]
listOfLists = []
listOfLists.append(first)
listOfLists.append(second)

manager = multiprocessing.Manager()
trained_models = manager.list()
counter = manager.Value('i', 0)


def fit(model):
    data = model.fit(disp=False)
    trained_models.append(data)
    counter.value += 1

    model_path = MODELS / f'model_{counter.value}'
    with open(model_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'finish {counter.value} / {len(first)}')


@timeit
def train_model():
    with multiprocessing.Pool(processes=CPU_COUNT) as mp:
        mp.map(fit, [model for model in first])


print(f'starting to train {len(first)} models')
train_model()
print(f'finished training')

#results = getResultes(models_SARIMAX)
# daysForcast = 20
# daysPredict = 100
# forecasts = []
# predictions = []
#
# for i in range(len(trained_models)):
#     predictions.append(predict(trained_models[i], daysPredict))
# for i in range(len(trained_models)):
#     forecasts.append(pd.DataFrame(forecast(trained_models[i], daysForcast)))
# #ResultsDf = pd.DataFrame([[1, 2], [3, 4]], columns=['id','sales'])
# k = 0
# for detail in ModelDetailsList:
#     is_store_nbr = Test['store_nbr'] == detail[0]
#     Filtered_Test_df = Test[is_store_nbr].copy()
#     single_product = Filtered_Test_df.loc[Filtered_Test_df['family'] == detail[1]]
#     Y = single_product.copy()
#     if k == 0:
#         temp = pd.concat([forecasts[k].transpose()], axis=1)
#     else:
#         temp = pd.concat([temp, forecasts[k].transpose()])
#     k = k+1
#
#
# idList = []
# valueList = []
# for column in temp.columns:
#     j = 0
#     for j in range(len(temp[column])):
#         is_store_nbr = Test['store_nbr'] == ModelDetailsList[j][0]
#         Filtered_by_store_df = Test[is_store_nbr].copy()
#         Filtered_by_family_df = Filtered_by_store_df.loc[Filtered_by_store_df['family'] == ModelDetailsList[j][1]]
#         Filtered_by_date_df = Filtered_by_family_df.filter(items=[column], axis=0)
#         id = Filtered_by_date_df['id'][0]
#         value = temp[column][j]
#         idList.append(id)
#         valueList.append(value)
#
# ResultsDf = pd.DataFrame({'id': idList, 'sales': valueList}).sort_values('id')
# #print(results[k].summary())
# print(forecasts)
# file_name = 'Results'
# ResultsDf.to_csv(file_name, sep='\t')