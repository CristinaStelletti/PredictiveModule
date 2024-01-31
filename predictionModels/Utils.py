import configparser
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as plotIO
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

plotIO.kaleido.scope.mathjax = None


def load_configs():
    print("Loading configurations...")
    configParser = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.properties')

    try:
        configParser.read(config_path)
    except FileNotFoundError:
        print("Error: Configuration file not found!")

    return configParser


def computing_shift(dataset, avg_field, index_field):
    columns = [avg_field, index_field]

    # Computing the correlation shift needed, between average and average index of contemporaneity
    array = [sm.tsa.stattools.ccf(dataset[index_field], dataset[avg_field], adjusted=False)]
    array2 = [sm.tsa.stattools.ccf(dataset[avg_field], dataset[index_field], adjusted=False)]
    max_correlation1 = "{:.3f}".format(np.max(np.abs(array)))
    max_correlation2 = "{:.3f}".format(np.max(np.abs(array2)))

    if max_correlation1 > max_correlation2:
        shift_optimale = np.argmax(np.abs(array))
        index = 0
    else:
        print(max_correlation2)
        shift_optimale = np.argmax(np.abs(array2))
        index = 1

    # Original dataframe copy
    shifted_dataset = dataset.copy()

    # Shifting specific column of 'shift_optimale' positions
    shifted_dataset[columns[index]] = shifted_dataset[columns[index]].shift(shift_optimale)
    shifted_dataset = shifted_dataset.dropna()
    shifted_dataset[columns[index]] = shifted_dataset[columns[index]].astype(int)

    return shifted_dataset


def extracting_train_test_sets(dataset, test_perc):

    train_set = dataset[:-int(len(dataset) * test_perc)]
    test_set = dataset[int(len(dataset) * (1-test_perc)):]

    return train_set, test_set


def split_sequence(dataset, n_steps):
    X, y = list(), list()
    for i in range(len(dataset)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(dataset) - 1:
            break
        # gather input and output parts of the pattern
        # assegno alle sequenze X tutte le righe da i a i + n_steps (e di tutte le colonne, necessario per il
        # caso multivariato) e alla sequenza di y la riga end_ix e solo il valore della media delle durate ossia la colonna 0
        seq_x, seq_y = dataset[i:end_ix, :], dataset[end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def plotting(y1, y2, prediction_period, title):

    x_ticks = np.arange(0, prediction_period)
    y1_flat = np.array(y1).flatten()
    y2_flat = np.array(y2).flatten()

    dataframe = pd.DataFrame({"X": x_ticks, "Test set reale": y1_flat, "Predizioni": y2_flat})

    fig = px.line(dataframe, x='X', y=dataframe.columns[1:3], markers=True, color_discrete_map={"Test set reale": "goldenrod", "Predizioni": "purple"})
    fig.update_layout(
        xaxis_title="Tempo",
        yaxis_title="Media",
        legend=dict(title=dict(text='Variabili')),
        font=dict(
            family="Computer Modern",
            size=12,
            color="RebeccaPurple"
        ),
        autosize=False,
        width=700,
        height=500,
        title={
            'text': f'{title}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16},
        },
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            gridcolor='#edede9',
            gridwidth=1,
            zerolinecolor='#edede9',
            zerolinewidth=2
        ),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=True
    )
    plotIO.write_image(fig, f'../plots/{title}.pdf')
    fig.show()


def global_plotting(y_train, y_test, y2, title):

    y1_flat = np.concatenate((np.array(y_train).flatten(), np.array(y_test).flatten()))
    x_ticks1 = np.arange(0, len(y1_flat))
    y2_flat = np.array(y2).flatten()
    x_ticks2 = np.arange(len(y1_flat)-len(y2_flat), len(y1_flat))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_ticks1, y=y1_flat, mode='lines+markers', name='Dati reali', line=dict(color='goldenrod')))
    fig.add_trace(go.Scatter(x=x_ticks2, y=y2_flat, mode='lines+markers', name='Predizioni', line=dict(color='purple')))

    fig.update_layout(
        xaxis_title="Tempo",
        yaxis_title="Media",
        legend=dict(title=dict(text='Variabili')),
        font=dict(
            family="Computer Modern",
            size=12,
            color="RebeccaPurple"
        ),
        autosize=False,
        width=1000,
        height=500,
        title={
            'text': f'{title}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16},
        },
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            gridcolor='#edede9',
            gridwidth=1,
            zerolinecolor='#edede9',
            zerolinewidth=2
        ),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=True
    )
    plotIO.write_image(fig, f'../plots/{title}.pdf')
    fig.show()


def show_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = [x for x in range(1, len(loss))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=loss, name="Training loss", mode="lines+markers", line=dict(color='goldenrod')))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Validation loss", mode="lines+markers", line=dict(color='purple')))
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend=dict(title=dict(text='Variabili')),
        font=dict(
            family="Computer Modern",
            size=12,
            color="RebeccaPurple"
        ),
        autosize=False,
        width=700,
        height=500,
        title={
            'text': f'{title}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16},
        },
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            gridcolor='#edede9',
            gridwidth=1,
            zerolinecolor='#edede9',
            zerolinewidth=2,
        ),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=True
    )

    #fig.write_html(f'../plots/{title}.html')
    plotIO.write_image(fig, f'../plots/{title}.pdf')
    fig.show()


def computing_metrics(avg_test, prediction_test):

    errore = np.mean(abs((avg_test - prediction_test)/avg_test))
    mape = errore*100
    mae = mean_absolute_error(avg_test, prediction_test)
    mse = mean_squared_error(avg_test, prediction_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(avg_test, prediction_test)
    return mse, rmse, mae, mape, r2


def saving_metrics(filepath, metrics_df, new_row):
    if os.path.isfile(filepath):
        # Se esiste, leggi il file CSV esistente
        metrics = pd.read_csv(filepath)
    else:
        # Se non esiste, crea un nuovo DataFrame vuoto
        metrics = metrics_df

    metrics.loc[len(metrics)] = new_row
    metrics.to_csv(filepath, index=False)
