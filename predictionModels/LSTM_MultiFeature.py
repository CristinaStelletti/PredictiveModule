import pandas as pd
import numpy as np
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense
import tensorflow as tf
import Utils
from sklearn.preprocessing import RobustScaler

configs = Utils.load_configs()
JUDGE_MOVING_AVG_FIELD = configs.get('JsonFields', 'judge.moving.average.by.object.field')
AVG_CONTEMPORANEITY_INDEX_FIELD = configs.get('JsonFields', 'average.contemporaneity.index.field')
FILE_PATH = configs.get('FilePaths', 'lstm.multivariate.metrics.filepath')
N_STEPS = int(configs.get('ModelParams', 'time.steps.LSTM'))


def runLSTMMultiFeature(dataset, shift, judge, judicial_object):
    tf.random.set_seed(42)

    # Shift della serie
    if shift:
        dataset = Utils.computing_shift(dataset, JUDGE_MOVING_AVG_FIELD, AVG_CONTEMPORANEITY_INDEX_FIELD)

    n_units = 168
    n_units2 = 48
    n_units3 = 24
    n_epochs = 200
    batch_size = 64
    activation_function = 'relu'
    loss_function = 'mae'
    opt = 'rmsprop'

    n_steps = N_STEPS  # il numero di valori passati che devo osservare per predire il valore futuro
    n_features = 2
    perc_test = 0.10

    scaler_duration = RobustScaler()
    scaler_IndiceContemporaneita = RobustScaler()

    serie_train, serie_test = Utils.extracting_train_test_sets(dataset, perc_test)

    X_train = scaler_IndiceContemporaneita.fit_transform(np.array(serie_train[AVG_CONTEMPORANEITY_INDEX_FIELD]).reshape(-1, 1))
    y_train = scaler_duration.fit_transform(np.array(serie_train[JUDGE_MOVING_AVG_FIELD]).reshape(-1, 1))

    X_test = scaler_IndiceContemporaneita.transform(np.array(serie_test[AVG_CONTEMPORANEITY_INDEX_FIELD]).reshape(-1, 1))
    y_test = scaler_duration.transform(np.array(serie_test[JUDGE_MOVING_AVG_FIELD]).reshape(-1, 1))

    # Ricompatto la serie unendo le due colonne
    serie_train = np.hstack((y_train, X_train))
    serie_test = np.hstack((y_test, X_test))

    print(serie_train.shape)
    X_train, y_train = Utils.split_sequence(serie_train, n_steps)
    print(X_train.shape)
    X_test, y_test = Utils.split_sequence(serie_test, n_steps)

    for i, x_seq in enumerate(X_train):
        print(f"X: {x_seq} -> y: {y_train[i]}")

    print(X_train.shape)
    model = Sequential()
    model.add(Input((n_steps, n_features), None))
    # model.add(LSTM(n_units, activation=activation_function, return_sequences=True))
    # model.add(LSTM(n_units2, activation=activation_function, return_sequences=False))
    model.add(LSTM(n_units, activation=activation_function, return_sequences=True))
    model.add(LSTM(n_units2, activation=activation_function, return_sequences=True))
    model.add(LSTM(n_units3, activation=activation_function, return_sequences=False))

    # model.add(LSTM(n_units, activation=activation_function))
    model.add(Dense(1))

    model.compile(optimizer=opt, loss=loss_function)

    model.summary()

    checkpoint_filepath = '../checkpoint_multifeatureLSTM/checkpoint.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                    monitor="val_loss",
                                                    save_best_only=True, verbose=0)

    # fit model
    history = model.fit(X_train, y_train, epochs=n_epochs, validation_split=0.10, verbose=1, batch_size=batch_size,
                        shuffle=False,
                        callbacks=[EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01), checkpoint])
    # patience -> numero consecutivo di epoche che non portano a miglioramenti per cui si deve fermare

    model.load_weights(checkpoint_filepath)
    Utils.show_loss(history, title=f"LSTM multivariato - giudice {judge} - materia {judicial_object} - Confronto tra la training e la validation loss")

    y_pred = model.predict(X_test, verbose=0)

    y_predicted = y_pred.flatten()

    y_train = scaler_duration.inverse_transform(np.array(y_train).reshape(-1, 1))
    y_test = scaler_duration.inverse_transform(np.array(y_test).reshape(-1, 1))
    y_predicted = scaler_duration.inverse_transform(np.array(y_predicted).reshape(-1, 1))

    y_predicted = np.array(y_predicted).flatten()

    print("Y test: ", list(np.array(y_test).flatten()))
    print("Y_predicted", list(np.array(y_predicted).flatten()))

    for i, x_seq in enumerate(X_test):
        print(f"input seq: {x_seq} -> y_actual: {y_test[i]} y_pred: {y_predicted[i]}")

    Utils.global_plotting(y_train, y_test, y_predicted, title=f"LSTM Multivariato - giudice {judge} - materia {judicial_object} - Confronto tra la serie globale e le predizioni")
    Utils.plotting(y_test, y_predicted, len(y_test), title=f"LSTM Multivariato - giudice {judge} - materia {judicial_object} - Confronto tra il test set e le predizioni")

    mse, rmse, mae, mape, r2 = Utils.computing_metrics(y_test, y_predicted)

    metrics = pd.DataFrame(columns=['Judge', 'Judicial Object', 'N_STEPS', 'MSE', 'RMSE', 'MAE', 'MAPE(%)', 'R2'])

    new_row = {'Judge': judge, 'Judicial Object': judicial_object, 'N_STEPS': n_steps, 'MSE': mse,
               'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape, 'R2': r2}

    Utils.saving_metrics(FILE_PATH, metrics, new_row)

    return y_predicted