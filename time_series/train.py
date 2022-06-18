# SARIMAX - From here
import multiprocessing
import os
import pickle
from pathlib import Path
import sys

import pandas as pd
import statsmodels.api as sm

from utils import timeit
from config import TRAINED_MODELS_PATH, MODEL_DETAILS_PATH, DATA_FILE_TRAIN, DATA_SETS_PATH, SARIMAX_MODELS_PATH


# from statsmodels.tsa.arima.model import ARIMA
# import statsmodels.tsa.api as smt
# register_matplotlib_converters()
# sns.set_style("darkgrid")

def fit(model):
    data = model.fit(disp=False)
    counter.value += 1
    current_counter = counter.get()
    model_path = TRAINED_MODELS_PATH / f'model_{current_counter}'
    with open(model_path, 'wb') as f:
        pickle.dump(data, f)

    print(f'finish {current_counter} / {len(first)}')


@timeit
def train_model():
    with multiprocessing.Pool(processes=CPU_COUNT) as mp:
        mp.map(fit, [model for model in first])


if __name__ == '__main__':

    with_mp = sys.argv[1]
    how_many = int(sys.argv[2])
    with_cached_model_metadata = sys.argv[3]
    CPU_COUNT = multiprocessing.cpu_count()

    print(f'with multiprocessing: {with_mp}')
    print(f'cores: {CPU_COUNT}')
    print(f'how many samples: {how_many}')
    print(f'with cached model metadata: {with_cached_model_metadata}')

    # Copy of dataset
    Train = pd.read_csv(DATA_FILE_TRAIN, parse_dates=["date"])
    Train = Train.set_index("date").to_period("D")
    stores = Train.store_nbr.unique()
    families = Train.family.unique()

    i = 0
    results = []
    LEN = 5
    model_trained = 0
    ModelDetailsList = []
    datasets = []
    models_SARIMAX = []

    if with_cached_model_metadata == 'True':
        print('using cached data sets')
        with open(DATA_SETS_PATH, 'rb') as f:
            datasets = pickle.load(f)

        print('using cached srimax')
        with open(SARIMAX_MODELS_PATH, 'rb') as f:
            models_SARIMAX = pickle.load(f)

    else:
        print('building model from data')

        for store in stores:
            is_store_nbr = Train['store_nbr'] == store
            Filtered_df = Train[is_store_nbr].copy()
            for family in families:
                single_test = Filtered_df.loc[Filtered_df['family'] == family]
                X = single_test.copy()
                X["dayofyear"] = X.index.dayofyear
                X["year"] = X.index.year
                models_SARIMAX.append(
                    sm.tsa.statespace.SARIMAX(X.sales, trend="c", order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)))
                model_trained += 1
                datasets.append(X.sales)
                ModelDetailsList.append((store, family))

        print('finished to build model for ModelDetailsList')
        with open(MODEL_DETAILS_PATH, 'wb') as f:
            pickle.dump(ModelDetailsList, f)

        print('finished to build model for data sets')
        with open(DATA_SETS_PATH, 'wb') as f:
            pickle.dump(datasets, f)

        print('finished to build model for srimax')
        with open(SARIMAX_MODELS_PATH, 'wb') as f:
            pickle.dump(models_SARIMAX, f)

        print('finished to build model from data')

    cores = 2
    j = 0
    first = models_SARIMAX[0:how_many]
    second = models_SARIMAX[101:200]
    listOfLists = [first, second]

    print(f'starting to train {len(first)} models')

    if with_mp == 'True':
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        train_model()
    else:
        counter = 0
        for model in first:
            model_path = TRAINED_MODELS_PATH / f'model_{counter}'

            if not Path(model_path).is_file():
                data = model.fit(disp=False)
                with open(model_path, 'wb') as f:
                    pickle.dump(data, f)
                counter += 1
                print(f'finish {counter} / {len(first)}')

    print(f'finished training')
