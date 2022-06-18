import os

from pathlib import Path

DATA_FILE_TRAIN = Path.cwd().parent / 'data' / 'train.csv'
DATA_FILE_TEST = Path.cwd().parent / 'data' / 'test.csv'
MODEL_DATA = Path.cwd().parent / 'model_data'
TRAINED_MODELS_PATH = Path.cwd().parent / 'models'
MODEL_DETAILS_PATH = MODEL_DATA / 'model_details_list.data'
DATA_SETS_PATH = MODEL_DATA / 'data_sets.data'
SARIMAX_MODELS_PATH = MODEL_DATA / 'srimax_models.data'

if not Path(TRAINED_MODELS_PATH).is_dir():
    os.mkdir(TRAINED_MODELS_PATH)

if not Path(MODEL_DATA).is_dir():
    os.mkdir(MODEL_DATA)
