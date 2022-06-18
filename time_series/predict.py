import sys
import os
import pickle

import pandas as pd

from config import TRAINED_MODELS_PATH, DATA_FILE_TEST, MODEL_DETAILS_PATH

daysForcast = 20
daysPredict = 100
forecasts = []
predictions = []
Test = pd.read_csv(DATA_FILE_TEST, parse_dates=["date"])
Test = Test.set_index("date").to_period("D")

def predict(res, days):
    pred = res.get_prediction(start=days, dynamic=False)
    pred_ci = pred.conf_int()


def forecast(res, days):
    # days = 16
    k = 2
    pred_uc = res.get_forecast(steps=1 * days)
    pred_ci = pred_uc.conf_int()
    return pred_uc.predicted_mean[4:]


if __name__ == '__main__':

    how_many = int(sys.argv[1])
    trained_models = []
    models_counter = 0

    with open(MODEL_DETAILS_PATH, 'rb') as f:
        ModelDetailsList = pickle.load(f)

    for trained_model in os.listdir(TRAINED_MODELS_PATH):
        model_path = TRAINED_MODELS_PATH / trained_model
        with open(model_path, 'rb') as f:
            trained_model_data = pickle.load(f)
        models_counter += 1
        if models_counter > how_many:
            break

        forecast = pd.DataFrame(forecast(trained_model_data, daysForcast))
        forecast_df = pd.concat([forecast.transpose()], axis=1)

        idList = []
        valueList = []
        for column in forecast_df.columns:
            j = 0
            for j in range(len(forecast_df[column])):
                is_store_nbr = Test['store_nbr'] == ModelDetailsList[j][0]
                Filtered_by_store_df = Test[is_store_nbr].copy()
                Filtered_by_family_df = Filtered_by_store_df.loc[Filtered_by_store_df['family'] == ModelDetailsList[j][1]]
                Filtered_by_date_df = Filtered_by_family_df.filter(items=[column], axis=0)
                id = Filtered_by_date_df['id'][0]
                value = forecast[column][j]
                idList.append(id)
                valueList.append(value)

        ResultsDf = pd.DataFrame({'id': idList, 'sales': valueList}).sort_values('id')
        # print(results[k].summary())
        # print(forecasts)
        file_name = 'Results'
        ResultsDf.to_csv(file_name, sep='\t')