import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import Utils

configs = Utils.load_configs()
JUDGE_MOVING_AVG_FIELD = configs.get('JsonFields', 'judge.moving.average.by.object.field')
AVG_CONTEMPORANEITY_INDEX_FIELD = configs.get('JsonFields', 'average.contemporaneity.index.field')
FILE_PATH = configs.get('FilePaths', 'arimax.metrics.filepath')


def splitting_dataset(averages, indexes, testSetDim):

    test_set = averages[-testSetDim:]
    train_set = averages[:-testSetDim]
    indexes_test = indexes[-testSetDim:]
    indexes_train = indexes[:-testSetDim]

    return np.array(train_set), np.array(test_set), np.array(indexes_train), np.array(indexes_test)


def predictions_and_metrics(judge, judicial_object, train_set, test_set, indexes_train, indexes_test, test_perc):
    # Auto-ARIMA Model
    inf_criterion = ['aic', 'aicc', 'bic', 'hqic']
    methods = ['lbfgs', 'bfgs', 'cg', 'nm', 'powell']
    mse_final = {}
    rmse_final = {}
    mae_final = {}
    mape_final = {}
    r2_final = {}
    model_order = {}
    all_predictions = {}
    min_mae = np.inf
    min_criterion = ''

    for criterion in inf_criterion:
        mse_final[criterion] = []
        rmse_final[criterion] = []
        mae_final[criterion] = []
        mape_final[criterion] = []
        r2_final[criterion] = []
        model_order[criterion] = []
        all_predictions[criterion] = []

        indexes_train = indexes_train.reshape(-1, 1)

        for method in methods:
            arima_model = auto_arima(
                train_set,
                exog=indexes_train,
                maxiter=100,
                n_fits=50,
                seasonal=False,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                method=method,
                information_criterion=criterion)

            p, d, q = arima_model.order
            model = ARIMA(train_set, exog=indexes_train, order=(p, d, q))

            # Adatta il modello ai dati
            fit_model = model.fit()

            # Visualizza un riassunto del modello
            print(fit_model.summary())
            prediction_values = fit_model.predict(start=len(train_set), end=len(train_set) + len(test_set) - 1, exog=indexes_test)

            prediction_values = np.array(prediction_values)
            test_set = np.array(test_set)

            if not all(elemento == 0 for elemento in prediction_values):
                mse, rmse, mae, mape, r2 = Utils.computing_metrics(test_set, prediction_values)
                model_order[criterion].append((p, d, q))
                mse_final[criterion].append(mse)
                rmse_final[criterion].append(rmse)
                mae_final[criterion].append(mae)
                mape_final[criterion].append(mape)
                r2_final[criterion].append(r2)
                all_predictions[criterion].append(prediction_values)

    for criterion in inf_criterion:
        if min_mae > min(mae_final[criterion]):
            min_mae = min(mae_final[criterion])
            min_criterion = criterion

    index = mae_final[min_criterion].index(min_mae)
    print(f"\n\nMigliori risultati raggiunti con l'information criterion {min_criterion} e il metodo {methods[index]}")
    Utils.plotting(test_set, all_predictions[min_criterion][index], len(test_set), title=f"ARIMAX - giudice {judge} - materia {judicial_object} - Confronto test set e predizioni")
    Utils.global_plotting(train_set, test_set, all_predictions[min_criterion][index], title=f"ARIMAX - giudice {judge} - materia {judicial_object} - Confronto serie globale e predizioni")

    metrics = pd.DataFrame(
        columns=['Judge', 'Judicial Object', '%TEST', 'Criterion', 'Method', 'Order', 'MSE', 'RMSE', 'MAE',
                 'MAPE(%)', 'R2'])

    new_row = {'Judge': judge, 'Judicial Object': judicial_object, '%TEST': int(test_perc * 100),
               'Criterion': min_criterion, 'Method': methods[index], 'Order': model_order[min_criterion][index],
               'MSE': mse_final[min_criterion][index], 'RMSE': rmse_final[min_criterion][index],
               'MAE': mae_final[min_criterion][index], 'MAPE(%)': mape_final[min_criterion][index],
               'R2': r2_final[min_criterion][index]}

    Utils.saving_metrics(FILE_PATH, metrics, new_row)

    return all_predictions[min_criterion][index]


def runARIMAX(dataset, judge, judicial_object, test_perc, shift):

    if shift:
        dataset = Utils.computing_shift(dataset, JUDGE_MOVING_AVG_FIELD, AVG_CONTEMPORANEITY_INDEX_FIELD)

    testSet_dim = int(len(dataset[JUDGE_MOVING_AVG_FIELD]) * test_perc)

    train_set, test_set, indexes_train, indexes_test = splitting_dataset(dataset[JUDGE_MOVING_AVG_FIELD], dataset[AVG_CONTEMPORANEITY_INDEX_FIELD], testSet_dim)

    predictions = predictions_and_metrics(judge, judicial_object, train_set, test_set, indexes_train, indexes_test, test_perc)

    return predictions
