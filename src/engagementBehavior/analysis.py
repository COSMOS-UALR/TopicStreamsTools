import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import data_pre_processing, train
from ..dataManager import load_df, save_df


"""# Rolling Window"""


def showRollingWindow(rolling_df, x, y):
    """Plot the rolling window graph."""
    _, ax = plt.subplots(figsize=(14, 6))
    rolling_df.plot(ax=ax)
    ax.set(xlabel='Frame', ylabel='Pearson r')
    ax.set_title(x + " vs " + y)
    plt.suptitle('Rolling Window Correlation')


def create_rolling_window_df(df, x, y):
    """Take timestamped x-y dataframe and append rolling window correlation and end dates."""
    date = df.columns.values[0]
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=100)
    rolling_df = df[x].rolling(window=indexer, min_periods=100).corr(df[y])
    # showRollingWindow(rolling_df, x, y)
    rolling_df = rolling_df.to_frame()
    rolling_df.columns = ['corr_value']
    out_df = pd.concat((df, rolling_df), axis=1)
    length = out_df.shape[0]
    end_dates_list = [out_df['date'][i + 99] if length - 99 > i else np.NaN for i in range(length)]
    out_df['end_date'] = end_dates_list
    out_df.dropna(how='any', inplace=True)
    correlation = out_df.columns.values[3]
    end_date = out_df.columns.values[4]
    cols = [date, end_date, x, y, correlation]
    out_df = out_df[cols].reset_index(drop=True)
    return out_df.replace([np.inf, -np.inf], np.nan).fillna(0)


"""# Anomaly Detection"""


def getAnomalyDF(corr_df, loss_df, threshold, start_date):
    """Given the correlation df & loss df, compute square of square errors, tag as anomalous if over threshold and after start_date (if non null), and filter out non anomalies."""
    df = corr_df.merge(loss_df, how='inner', left_on=['date'], right_on=['date'])
    df['sse'] = df['corr_value'].apply(lambda x: (1 - x) ** 2)
    if start_date:
        anomaly_list = [1 if x[6] > float(threshold) and x[2] > start_date else 0 for x in df.itertuples()]
    else:
        anomaly_list = [1 if x[6] > float(threshold) else 0 for x in df.itertuples()]
    df['is_anomaly'] = anomaly_list
    df.sort_values('is_anomaly', ascending=False, inplace=True)
    df.where(df['is_anomaly'] == 1, inplace=True)
    df.dropna(how='all', inplace=True)
    df.sort_values('date', ascending=True, inplace=True)
    return df


def getLossDF(settings, anomaly_type, channel_id, df):
    """For the given channel_id's df, will attempt to load the loss dataframe, or will retrain a model."""
    loss_df_file = f"{channel_id}_{anomaly_type}.pkl"
    try:
        loss_df = load_df(settings, loss_df_file)
    except FileNotFoundError:
        loss_df = None
    if loss_df is None or ('retrain' in settings and settings['retrain']):
        X, Y, T = data_pre_processing(df, settings['lookback_size'])
        loss = train(X, Y, settings['model'], anomaly_type)
        # Adjust index and date column for later plotting
        loss_df = pd.DataFrame(loss, columns=["loss"], index=pd.to_datetime(T))
        loss_df["date"] = pd.to_datetime(T)
        save_df(settings, loss_df_file, loss_df)
    return loss_df


"""# Anomaly Aggregation"""


def getIndicesList(df, time_frame):
    """Based on the given time frame to aggregate on, return tuple of the df with format '(starting index, ending index, duration as string)'."""
    indices_list = []
    start_date_index = 0
    end_date_index = 0
    for i in range(1, len(df)):
        duration = df['date'][i] - df['date'][start_date_index]
        if duration < time_frame:
            end_date_index = i
        else:
            indices_list.append((start_date_index, end_date_index,
                str(df['end_date'][end_date_index] - df['date'][start_date_index])
                .replace(' 00:00:00', "")))
            start_date_index = i
            end_date_index = i
    indices_list.append((start_date_index, end_date_index,
        str(df['end_date'][end_date_index] - df['date'][start_date_index])
        .replace(' 00:00:00', "")))
    return indices_list


def buildAnomalyStats(anomaly_df, x, y, time_frame):
    """Based on the given time frame, compute average values for the anomaly df."""
    anomaly_df.reset_index(inplace=True, drop=True)
    indices_list = getIndicesList(anomaly_df, time_frame)
    col_a_name = x.split('_')[1]
    col_b_name = y.split('_')[1]
    anomaly_df_list = []
    for item in indices_list:
        start_date = anomaly_df['date'][item[0]]
        end_date = anomaly_df['end_date'][item[1]]
        duration = item[2]
        avg_a = int(np.average(anomaly_df[x][item[0]:item[1] + 1]))
        avg_b = int(np.average(anomaly_df[y][item[0]:item[1] + 1]))
        min_corr = min(anomaly_df['corr_value'][item[0]:item[1] + 1])
        max_score = max(anomaly_df['loss'][item[0]:item[1] + 1])
        avg_sse = math.sqrt(np.sum(anomaly_df['sse'][item[0]:item[1] + 1])) / len(
            anomaly_df['sse'][item[0]:item[1] + 1])
        anomaly_df_list.append((start_date, end_date, duration, avg_a, avg_b, min_corr, max_score, avg_sse))
    return pd.DataFrame(anomaly_df_list, columns=['start_date', 'end_date',
        'duration', 'avg_' + col_a_name, 'avg_' + col_b_name, 'min_corr', 'max_anomaly_score', 'avg_sse'])


"""# Combine Anomalies"""


def transform_anomaly_output(aggregated_anomalies, anomaly_type, channel_id):
    """Rename columns, format duration, and add channel id to df."""
    col_headers = aggregated_anomalies.columns.values
    aggregated_anomalies['duration'] = aggregated_anomalies['duration'].apply(lambda x: int(x.split(" ")[0]))
    aggregated_anomalies = aggregated_anomalies.rename(columns=
        {
            'duration': "duration(in days)",
            'avg_corr': "avg_corr(" + anomaly_type + ")",
            'avg_anomaly_score': f"avg_anomaly_score({anomaly_type})",
            'avg_sse': f"avg_sse({anomaly_type})",
            col_headers[3]: f"{col_headers[3]}({anomaly_type})",
            col_headers[4]: f"{col_headers[4]}({anomaly_type})"
        })
    aggregated_anomalies.insert(0, 'channel_id', channel_id)
    return aggregated_anomalies
