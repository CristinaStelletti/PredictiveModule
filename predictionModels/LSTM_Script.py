import numpy as np
import pandas as pd
from keras import Input
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import Utils

configs = Utils.load_configs()

JUDGE_MOVING_AVG_FIELD = configs.get('JsonFields', 'judge.moving.average.by.object.field')
AVG_CONTEMPORANEITY_INDEX_FIELD = configs.get('JsonFields', 'average.contemporaneity.index.field')
FILE_PATH = configs.get('FilePaths', 'lstm.univariate.metrics.filepath')
N_STEPS = int(configs.get('ModelParams', 'time.steps.LSTM'))


def runLSTM(dataset, judge, judicial_object):
    np.random.seed(42)
    tf.random.set_seed(42)

    data = dataset[JUDGE_MOVING_AVG_FIELD]
    print(len(data))

    n_units = 50
    n_units2 = 48
    n_epochs = 100
    n_features = 1
    batch_size = 64
    n_steps = N_STEPS  # il numero di valori passati che devo osservare per predire il valore futuro
    activation_function = 'relu'
    loss_function = 'mae'
    opt = 'nadam'

    perc_test = 0.10

    scaler = RobustScaler()

    serie_train, serie_test = Utils.extracting_train_test_sets(data, perc_test)

    serie_train = np.array(serie_train).reshape(-1, 1)
    serie_test = np.array(serie_test).reshape(-1, 1)

    serie_train = scaler.fit_transform(serie_train)
    serie_test = scaler.transform(serie_test)

    X_train, y_train = Utils.split_sequence(serie_train, n_steps)
    X_test, y_test = Utils.split_sequence(serie_test, n_steps)

    for i in range(X_train.shape[0]):
        print(f"X: {X_train[i]} -> y: {y_train[i]}")

    model = Sequential()
    model.add(Input((n_steps, n_features), None))
    model.add(LSTM(n_units, activation=activation_function))  # , return_sequences=True))
    # model.add(LSTM(n_units, activation=activation_function, input_shape=(n_steps, n_features), return_sequences=True))
    # model.add(LSTM(n_units2, activation=activation_function, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer=opt, loss=loss_function)

    model.summary()

    checkpoint_filepath = '../checkpoint_univariateLSTM/checkpoint.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                    monitor="val_loss",
                                                    save_best_only=True, verbose=0)

    # fit model
    history = model.fit(X_train, y_train, epochs=n_epochs, validation_split=0.10, verbose=1, batch_size=batch_size,
                        shuffle=False,
                        callbacks=[EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01), checkpoint])
    # patience -> numero consecutivo di epoche che non portano a miglioramenti per cui si deve fermare

    model.load_weights(checkpoint_filepath)
    Utils.show_loss(history, f"Modello LSTM Univariate - giudice {judge} - materia {judicial_object} - Confronto tra training e validation loss")

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
