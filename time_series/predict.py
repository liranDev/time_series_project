import sys
import os
import pickle

import pandas as pd

from config import TRAINED_MODELS_PATH, DATA_FILE_TEST

daysForcast = 20
daysPredict = 100
forecasts = []
predictions = []
Test = pd.read_csv(DATA_FILE_TEST, parse_dates=["date"])
Test = Test.set_index("date").to_period("D")


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

    for trained_model in os.listdir(TRAINED_MODELS_PATH):
        model_path = TRAINED_MODELS_PATH / trained_model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            trained_model_data = model['train_data']
            model_detail_list = model['model_detail_list']
        models_counter += 1
        if models_counter > how_many:
            break

        forecast = pd.DataFrame(forecast(trained_model_data, daysForcast))
        forecast_df = pd.concat([forecast.transpose()], axis=1)

        idList = []
        valueList = []
        for column in forecast_df.columns:
            is_store_nbr = Test['store_nbr'] == model_detail_list[0]
            Filtered_by_store_df = Test[is_store_nbr].copy()
            Filtered_by_family_df = Filtered_by_store_df.loc[Filtered_by_store_df['family'] == model_detail_list[1]]
            Filtered_by_date_df = Filtered_by_family_df.filter(items=[column], axis=0)
            id = Filtered_by_date_df['id']
            value = forecast_df[column]
            idList.append(id)
            valueList.append(value)

        ResultsDf = pd.DataFrame({'id': idList, 'sales': valueList})
        file_name = f'model_results_{model_detail_list[0]}_{model_detail_list[1]}'
        ResultsDf.to_csv(file_name, sep='\t')
