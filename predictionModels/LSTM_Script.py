import os

import keras.models
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

import Utils

configs = Utils.load_configs()

JUDGE_MOVING_AVG_FIELD = configs.get('JsonFields', 'judge.moving.average.by.object.field')
FILE_PATH = configs.get('FilePaths', 'lstm.univariate.metrics.filepath')
N_STEPS = int(configs.get('ModelParams', 'time.steps.LSTM'))
CHECKPOINTING_DIR = configs.get('FilePaths', 'lstm.univariate.checkpointing.dir')


def runLSTM(dataset, judge, judicial_object):
    np.random.seed(42)
    tf.random.set_seed(42)

    data = dataset[JUDGE_MOVING_AVG_FIELD]

    perc_test = 0.10
    n_steps = N_STEPS

    scaler = RobustScaler()

    serie_train, serie_test = Utils.extracting_train_test_sets(data, perc_test)

    serie_train = np.array(serie_train).reshape(-1, 1)
    serie_test = np.array(serie_test).reshape(-1, 1)

    serie_train = scaler.fit_transform(serie_train)
    serie_test = scaler.transform(serie_test)

    X_train, y_train = Utils.split_sequence(serie_train, n_steps)
    X_test, y_test = Utils.split_sequence(serie_test, n_steps)

    files = os.listdir(CHECKPOINTING_DIR)
    model_filepath = CHECKPOINTING_DIR+'/'+files[0]

    model = keras.models.load_model(model_filepath, compile=False)

    y_pred = model.predict(X_test, verbose=0)

    y_predicted = np.array([x[0] for x in y_pred])

    y_train = scaler.inverse_transform(np.array(y_train).reshape((-1, 1)))
    y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    y_predicted = scaler.inverse_transform(np.array(y_predicted).reshape(-1, 1))
    y_predicted = np.array(y_predicted).flatten()

    for i in range(X_test.shape[0]):
        print(f"input seq: {X_test[i]} -> y_actual: {y_test[i]} y_pred: {y_predicted[i]}")

    Utils.global_plotting(y_train, y_test, y_predicted, title=f"LSTM Univariato - giudice {judge} - materia {judicial_object} - Confronto globale tra la serie reale e le predizioni")
    Utils.plotting(y_test, y_predicted, len(y_test), title=f"LSTM Univariato - giudice {judge} - materia {judicial_object} - Confronto tra il test set e le predizioni")

    mse, rmse, mae, mape, r2 = Utils.computing_metrics(y_test, y_predicted)

    metrics = pd.DataFrame(columns=['Judge', 'Judicial Object', 'N_STEPS', 'MSE', 'RMSE', 'MAE', 'MAPE(%)', 'R2'])

    new_row = {'Judge': judge, 'Judicial Object': judicial_object, 'N_STEPS': n_steps, 'MSE': mse, 'RMSE': rmse, 'MAE': mae,
               'MAPE(%)': mape, 'R2': r2}

    Utils.saving_metrics(FILE_PATH, metrics, new_row)

    return y_predicted
