import csv
import subprocess
import sys
from datetime import timedelta
from pymongo import *
import ARIMAX_Script, LSTM_MultiFeature, LSTM_Script, Utils
import pandas as pd

print("Loading configurations...")
configs = Utils.load_configs()

# Load MongoDB config
CLIENT = configs.get('MongoDB', 'mongo.client')
DB_NAME = configs.get('MongoDB', 'db.name')
AVG_BY_OBJECT_COLLECTION_NAME = configs.get('MongoDB', 'average.judge.by.object.collection')
FORECAST_BY_OBJECT_COLLECTION = configs.get('MongoDB', 'forecast.average.by.object.collection')

# Load JsonField config
JUDICIAL_OFFICE_FIELD = configs.get('JsonFields', 'judicial.office.field')
JUDGE_FIELD = configs.get('JsonFields', 'judge.field')
OBJECT_FIELD = configs.get('JsonFields', 'object.field')
SECTION_FIELD = configs.get('JsonFields', 'section.field')
END_DATE_FIELD = configs.get('JsonFields', 'end.date.field')
JUDGE_MOVING_AVG_FIELD = configs.get('JsonFields', 'judge.moving.average.by.object.field')
AVG_CONTEMPORANEITY_INDEX_FIELD = configs.get('JsonFields', 'average.contemporaneity.index.field')

# Load ModelParams config
PREDICTIVE_MODEL = configs.get('ModelParams', 'predictive.model')
SHIFT = bool(configs.get('ModelParams', 'shift.option'))
N_STEPS = int(configs.get('ModelParams', 'time.steps.LSTM'))

# Load PredictionParams config
TEST_PERC_ARIMAX = float(configs.get('PredictionParams', 'test.percentage.ARIMAX'))
TEST_PERC_LSTM = float(configs.get('PredictionParams', 'test.percentage.LSTM'))
SAMPLING_PERIOD = int(configs.get('PredictionParams', 'sampling.period.days'))
MIN_OBS = int(configs.get('PredictionParams', 'minimum.number.observations'))


def create_json_with_prediction(predictions, dataset, judicial_object, test_perc):
    new_json_list = []
    try:
        if PREDICTIVE_MODEL == "ARIMAX":
            dataset_test = dataset[-(int(len(dataset)*(test_perc))+1):]
            data_prediction = pd.to_datetime(dataset_test[END_DATE_FIELD].iloc[0])
        else:
            dataset_test = dataset[-(int(len(dataset) * (test_perc)) + 1 - N_STEPS):]
            data_prediction = pd.to_datetime(dataset_test[END_DATE_FIELD].iloc[0])

        for element in predictions:
            # Aggiungi x giorni alla data di previsione
            data_prediction = pd.to_datetime(data_prediction) + timedelta(SAMPLING_PERIOD)
            data_prediction_str = data_prediction.strftime('%Y-%m-%d')

            json_data = {
                JUDICIAL_OFFICE_FIELD: dataset[JUDICIAL_OFFICE_FIELD][0],
                JUDGE_FIELD: str(dataset[JUDGE_FIELD][0]),
                OBJECT_FIELD: str(judicial_object),
                SECTION_FIELD: dataset[SECTION_FIELD][0],
                JUDGE_MOVING_AVG_FIELD: float(element),
                END_DATE_FIELD: data_prediction_str
            }

            new_json_list.append(json_data)
    except Exception as e:
        print('Errore imprevisto in create_json_with_prediction: ', e)

    return new_json_list


def computing_and_saving_predictions(filepath, dataset, judge, judicial_object, forecast_collection):
    print(f"...Model creation for judicial object:{judicial_object} and for judge: {judge}")
    predictions = None
    TEST_PERC = 0
    args = [filepath, judge, judicial_object]
    if PREDICTIVE_MODEL == 'ARIMAX':
        TEST_PERC = TEST_PERC_ARIMAX
        predictions = ARIMAX_Script.runARIMAX(dataset, judge, judicial_object, TEST_PERC_ARIMAX, SHIFT)
    elif PREDICTIVE_MODEL == 'LSTM_Univariate':
        TEST_PERC = TEST_PERC_LSTM
        subprocess.run([sys.executable, "UnivariateLSTM_optimization.py"] + args)
        predictions = LSTM_Script.runLSTM(dataset, judge, judicial_object)
    elif PREDICTIVE_MODEL == 'LSTM_Multivariate':
        TEST_PERC = TEST_PERC_LSTM
        subprocess.run([sys.executable, "MultivariateLSTM_optimization.py"] + args)
        predictions = LSTM_MultiFeature.runLSTMMultiFeature(dataset, SHIFT, judge, judicial_object)

    if predictions is not None:
        try:
            new_json_list = create_json_with_prediction(predictions, dataset, judicial_object, TEST_PERC)
            print("Json list creation done. Loading data on db...")

            # Controllo inesistenza del dato e scrittura nel DB
            for elem in new_json_list:
                query = {JUDGE_FIELD: elem[JUDGE_FIELD], OBJECT_FIELD: judicial_object, END_DATE_FIELD: elem[END_DATE_FIELD]}
                cursor = forecast_collection.find(query)
                document = list(cursor)
                if not document:
                    forecast_collection.insert_one(elem)
                else:
                    document_filter = {"_id": document[0]["_id"]}

                    # Update instruction
                    update = {"$set": {JUDGE_MOVING_AVG_FIELD: elem[JUDGE_MOVING_AVG_FIELD]}}

                    forecast_collection.update_one(document_filter, update)
            print("Writing to db completed")
        except Exception as e:
            print("Computing and saving prediction error: ", e)


def processing(avg_collection, forecast_collection):
    try:
        query = {JUDGE_FIELD: {"$exists": True}}

        cursor = avg_collection.find(query)

        judges = cursor.distinct(JUDGE_FIELD)
        for judge in judges:
            query = {JUDGE_FIELD: judge}
            cursor = avg_collection.find(query)
            objects = cursor.distinct(OBJECT_FIELD)
            for judicial_object in objects:
                query2 = {OBJECT_FIELD: judicial_object}
                combined_query = {'$and': [query, query2]}
                result = list(avg_collection.find(combined_query).sort(END_DATE_FIELD, 1))
                title = result[0].keys()
                if len(result) > MIN_OBS:
                    with open(f'../data/avg_indexes_{judicial_object}_{judge}.csv', "w", newline="", encoding='utf-8') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(title)
                        for doc in result:
                            csv_writer.writerow(doc.values())
                    filepath = f'../data/avg_indexes_{judicial_object}_{judge}.csv'
                    dataset = pd.read_csv(filepath)
                    computing_and_saving_predictions(filepath, dataset, judge, judicial_object, forecast_collection)
                else:
                    print("Not enough averages to start forecasting")
    except Exception as e:
      print(f"Error: {e}")


def db_connection():
    try:
        client = MongoClient(CLIENT)
        db = client[DB_NAME]
        avg_by_object_collection = db[AVG_BY_OBJECT_COLLECTION_NAME]
        forecast_by_object_collection = db[FORECAST_BY_OBJECT_COLLECTION]

        return avg_by_object_collection, forecast_by_object_collection

    except Exception as e:
        print(f"Errore nella connessione al db: {e}")


if __name__ == '__main__':
    avg_by_object_collection, forecast_by_object_collection = db_connection()
    print("Estrazione dati e filtraggio...")
    processing(avg_by_object_collection, forecast_by_object_collection)
