import sys
import optuna
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from keras import Input, Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from keras.layers import LSTM, Dense, Bidirectional
import Utils

configs = Utils.load_configs()

JUDGE_MOVING_AVG_FIELD = configs.get('JsonFields', 'judge.moving.average.by.object.field')
AVG_CONTEMPORANEITY_INDEX_FIELD = configs.get('JsonFields', 'average.contemporaneity.index.field')
N_STEPS = int(configs.get('ModelParams', 'time.steps.LSTM'))
CHECKPOINTING_DIR = configs.get('FilePaths', 'lstm.multivariate.checkpointing.dir')
SHIFT = configs.get('ModelParams', 'shift.option')

# SET UP ENVIRONMENT
tf.random.set_seed(42)


def objective(trial):

    # I TRIED DIFFERENT MODELS CONFIGURATION, VARYING LSTM LAYERS AND NEURONS PER LAYER...
    models = dict()
    models[1] = [168, 48, 24]
    models[2] = [168, 48]
    models[3] = [64]
    models[4] = [64, 24]
    models[5] = [50]
    models[6] = [144, 64]
    models[7] = [144]

    idx = trial.suggest_categorical("model", [x for x in range(1, len(models))])
    # TRY USING BIDIRECTIONAL WRAPPER
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])

    # SELECT BATCH SIZE
    batch_sizes = [12, 32, 64]
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)

    epochs = [100, 150, 200]
    n_epochs = trial.suggest_categorical('n_epochs', epochs)

    lstm_activation = trial.suggest_categorical("lstm_act", ['relu', 'tanh'])

    opt = trial.suggest_categorical("opt", ['nadam', 'rmsprop'])

    model = Sequential()
    model.add(Input((N_STEPS, n_features), None))
    for unit in models[idx]:
        index = models[idx].index(unit)
        print("Index: ", index)
        if index != len(models[idx])-1:
            if bidirectional:
                model.add(Bidirectional(LSTM(unit, return_sequences=True, activation=lstm_activation)))
            else:
                model.add(LSTM(unit, return_sequences=True, activation=lstm_activation))
        else:
            if bidirectional:
                model.add(Bidirectional(LSTM(unit, return_sequences=False, activation=lstm_activation)))
            else:
                model.add(LSTM(unit, return_sequences=False, activation=lstm_activation))

    # TIME DISTRIBUTED LAYER SIMPY REPLIES THE LAST LSTM LAYER OUTPUTS AND APPLY TO EACH THE DENSE ACTIVATION FUNCTION
    # The TimeDistributed achieves this trick by applying the same Dense layer (same weights) to the LSTMs outputs for
    # one time step at a time. In this way, the output layer only needs one connection to each LSTM unit (plus one bias).
    # For this reason, the number of training epochs needs to be increased to account for the smaller network capacity.
    # I doubled it from 500 to 1000 to match the first one-to-one example.

    model.add(Dense(1))
    model.compile(optimizer=opt, loss=loss_function)

    model.summary()

    checkpoint_filepath = CHECKPOINTING_DIR+f'/checkpoint_{trial.number}.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                    monitor="val_loss",
                                                    save_best_only=True, verbose=0)

    # fit model
    history = model.fit(X_train, y_train, epochs=n_epochs, validation_split=0.10, verbose=1, batch_size=batch_size,
                        shuffle=False,
                        callbacks=[EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01), checkpoint])
    # patience -> numero consecutivo di epoche che non portano a miglioramenti per cui si deve fermare

    val_loss = history.history["val_loss"]
    train_loss = history.history['loss']
    trial.set_user_attr('train_loss', train_loss)
    trial.set_user_attr('val_loss', val_loss)

    model.load_weights(checkpoint_filepath)

    y_pred = model.predict(X_test, verbose=0)
    y_predicted = y_pred.flatten()

    y_true = scaler_duration.inverse_transform(np.array(y_test).reshape(-1, 1))
    y_predicted = scaler_duration.inverse_transform(np.array(y_predicted).reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_predicted)

    return mae


if __name__ == '__main__':

    file_path = sys.argv[1]
    judge = sys.argv[2]
    judicial_object = sys.argv[3]

    dataset = pd.read_csv(file_path)
    data = dataset[JUDGE_MOVING_AVG_FIELD]

    # Shift della serie
    if SHIFT:
        dataset = Utils.computing_shift(dataset, JUDGE_MOVING_AVG_FIELD, AVG_CONTEMPORANEITY_INDEX_FIELD)

    n_features = 2
    perc_test = 0.10
    loss_function = 'mae'

    scaler_duration = RobustScaler()
    scaler_IndiceContemporaneita = RobustScaler()

    serie_train, serie_test = Utils.extracting_train_test_sets(dataset, perc_test)

    X_train = scaler_IndiceContemporaneita.fit_transform(
        np.array(serie_train[AVG_CONTEMPORANEITY_INDEX_FIELD]).reshape(-1, 1))
    y_train = scaler_duration.fit_transform(np.array(serie_train[JUDGE_MOVING_AVG_FIELD]).reshape(-1, 1))

    X_test = scaler_IndiceContemporaneita.transform(
        np.array(serie_test[AVG_CONTEMPORANEITY_INDEX_FIELD]).reshape(-1, 1))
    y_test = scaler_duration.transform(np.array(serie_test[JUDGE_MOVING_AVG_FIELD]).reshape(-1, 1))

    # Ricompatto la serie unendo le due colonne
    serie_train = np.hstack((y_train, X_train))
    serie_test = np.hstack((y_test, X_test))

    X_train, y_train = Utils.split_sequence(serie_train, N_STEPS)
    X_test, y_test = Utils.split_sequence(serie_test, N_STEPS)

    print(f'Tensorflow recognized devices: {tf.config.experimental.list_physical_devices()}')

    print(f'Tensorflow recognize cuda: {tf.test.is_built_with_cuda()}')

    with tf.device('/gpu:0'):
        study = optuna.create_study(direction="minimize")
        # n_jobs = NUMBER OF PARALLEL  (python threads), for sequence configuration testing set to 1
        # n_trials = NUMBER OF COMBINATIONS TO TEST. IT MAY BE EQUAL TO EVERY POSSIBLE COMBINATIONS OF YOUR PARAMETERS
        # OR LOWER (SUBOPTIMAL SOLUTION)
        study.optimize(objective, n_jobs=2, show_progress_bar=True, n_trials=100, gc_after_trial=True)

    filename = f'checkpoint_{study.best_trial.number}.hdf5'
    training_loss = study.best_trial.user_attrs['train_loss']
    validation_loss = study.best_trial.user_attrs['val_loss']
    Utils.show_loss(training_loss, validation_loss, f"LSTM Multivariato - Giudice {judge} materia {judicial_object} - Confronto train e validation loss")
    Utils.clean_dir(CHECKPOINTING_DIR, filename)
