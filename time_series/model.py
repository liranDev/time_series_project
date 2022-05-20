import pandas as pd
import statsmodels.api as sm
from pathlib import Path


import multiprocessing

from utils import timeit


CPU_COUNT = multiprocessing.cpu_count()
DATA_FILE = Path.cwd().parent / 'data' / 'train.csv'

train = pd.read_csv(DATA_FILE, parse_dates=["date"])
train = train.set_index("date").to_period("D")

stores = train.store_nbr.unique()
families = train.family.unique()




i = 0
datasets = []
models_SARIMAX = []
results = []
LEN = 5
model_trained = 0
for store in stores:
    is_store_nbr = train['store_nbr'] == store
    Filtered_df = train[is_store_nbr].copy()
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
    break

cores = 2
j = 0
first = models_SARIMAX[0:100]
print(len(first))
second = models_SARIMAX[101:200]
# listOfLists.append(second)


manager = multiprocessing.Manager()

trained_models = manager.list()


def fit(model):

    data = model.fit(disp=False)
    trained_models.append(data)
    print('finish')


@timeit
def train_model():
    with multiprocessing.Pool(processes=CPU_COUNT) as mp:
        mp.map(fit, [model for model in first])


train_model()

print('finish all models !')

for model in trained_models:
    print('...')


# for i in range(cores):
#     Processes = []
#     #p = multiprocessing.Process(target=getResultes, args=models_SARIMAX[j:round(i*len(models_SARIMAX)/cores)])
#     p = multiprocessing.Process(target=getResultes, args=listOfLists[i])
#     j = j + len(models_SARIMAX)/cores
#     Processes.append(p)
#     p.start()