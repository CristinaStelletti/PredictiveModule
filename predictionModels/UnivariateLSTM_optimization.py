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

# SET UP ENVIRONMENT
tf.random.set_seed(42)

configs = Utils.load_configs()

JUDGE_MOVING_AVG_FIELD = configs.get('JsonFields', 'judge.moving.average.by.object.field')
N_STEPS = int(configs.get('ModelParams', 'time.steps.LSTM'))
CHECKPOINTING_DIR = configs.get('FilePaths', 'lstm.univariate.checkpointing.dir')


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
    # SELECT ACTIVATION FUNCTION (CHANGE TO SUIT YOUR PROBLEM)
    lstm_activation = trial.suggest_categorical("lstm_act", ['relu', 'tanh'])

    # SELECT BATCH SIZE
    batch_sizes = [12, 32, 64]
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)

    epochs = [100, 150, 200]
    n_epochs = trial.suggest_categorical('n_epochs', epochs)

    opt = trial.suggest_categorical("opt", ['nadam', 'rmsprop'])

    model = Sequential()
    model.add(Input((n_steps, n_features), None))
    for unit in models[idx]:
        index = models[idx].index(unit)
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

    model.add(Dense(1))
    model.compile(optimizer=opt, loss=loss_function)

    model.summary()

    checkpoint_filepath = CHECKPOINTING_DIR+f'/checkpoint_{trial.number}.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                    monitor="val_loss",
                                                    save_best_only=True, verbose=0, save_weights_only=False)

    # fit model
    model.fit(X_train, y_train, epochs=n_epochs, validation_split=0.10, verbose=1, batch_size=batch_size,
                        shuffle=False,
                        callbacks=[EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01), checkpoint])
    # patience -> numero consecutivo di epoche che non portano a miglioramenti per cui si deve fermare

    model.load_weights(checkpoint_filepath)

    y_pred = model.predict(X_test, verbose=0)

    y_predicted = np.array([x[0] for x in y_pred])

    y_true = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    y_predicted = scaler.inverse_transform(np.array(y_predicted).reshape(-1, 1))
    mae = mean_absolute_error(y_true, y_predicted)

    # MODELS CAN BE COMPARED USING JUST ONE METRIC, CHOOSE WHAT SUITS BETTER YOUR PROBLEM
    return mae


if __name__ == '__main__':

    file_path = sys.argv[1]

    dataset = pd.read_csv(file_path)
    data = dataset[JUDGE_MOVING_AVG_FIELD]

    n_features = 1
    n_steps = N_STEPS  # il numero di valori passati che devo osservare per predire il valore futuro
    loss_function = 'mae'

    perc_test = 0.10

    scaler = RobustScaler()

    serie_train, serie_test = Utils.extracting_train_test_sets(data, perc_test)

    serie_train = np.array(serie_train).reshape(-1, 1)
    serie_test = np.array(serie_test).reshape(-1, 1)

    serie_train = scaler.fit_transform(serie_train)
    serie_test = scaler.transform(serie_test)

    X_train, y_train = Utils.split_sequence(serie_train, n_steps)
    X_test, y_test = Utils.split_sequence(serie_test, n_steps)

    print(f'Tensorflow recognized devices: {tf.config.experimental.list_physical_devices()}')
    print(f'Tensorflow recognize cuda: {tf.test.is_built_with_cuda()}')

    with tf.device('/gpu:0'):
        study = optuna.create_study(direction="minimize")
        # n_jobs = NUMBER OF PARALLEL  (python threads), for sequence configuration testing set to 1
        # n_trials = NUMBER OF COMBINATIONS TO TEST. IT MAY BE EQUAL TO EVERY POSSIBLE COMBINATIONS OF YOUR PARAMETERS
        # OR LOWER (SUBOPTIMAL SOLUTION)
        study.optimize(objective, n_jobs=2, show_progress_bar=True, n_trials=100, gc_after_trial=True)

    filename = f'checkpoint_{study.best_trial.number}.hdf5'
    Utils.clean_dir(CHECKPOINTING_DIR, filename)
