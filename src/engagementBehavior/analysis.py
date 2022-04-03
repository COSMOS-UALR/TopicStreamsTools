from datetime import timedelta
import humanize
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm


"""# Rolling Window"""


def showRollingWindow(rolling_df, x, y):
    """Plot the rolling window graph."""
    _, ax = plt.subplots(figsize=(14, 6))
    rolling_df.plot(ax=ax)
    ax.set(xlabel='Frame', ylabel='Pearson r')
    ax.set_title(x + " vs " + y)
    plt.suptitle('Rolling Window Correlation')


def create_rolling_window_df(df):
    """Take timestamped x-y dataframe and rolling window correlation and end dates."""
    date = df.columns.values[0]
    x = df.columns.values[1]
    y = df.columns.values[2]
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


def trunc_data(df):
    a = df.columns.values[1]
    b = df.columns.values[2]
    c = df.columns.values[3]
    out_data = df.drop([a, b, c], axis=1)
    return out_data


def compute_and_visualize_anomalies(settings, df_totals, anomaly_type):
    in_data = trunc_data(df_totals)
    data, _ = read_modulate_data(in_data)
    X, Y, T = data_pre_processing(data, settings['lookback_size'])
    loss = train(X, Y, settings['model'], anomaly_type)
    loss_df = pd.DataFrame(loss, columns=["loss"])
    loss_df.index = T
    loss_df.index = pd.to_datetime(loss_df.index)
    loss_df["date"] = T
    loss_df["date"] = pd.to_datetime(loss_df["date"])
    return loss_df


def read_modulate_data(dataframe):
    """Data ingestion - Read and formulate the data."""
    dataframe.fillna(dataframe.mean(numeric_only=True), inplace=True)
    df = dataframe.copy()
    dataframe.set_index("date", inplace=True)
    dataframe.index = pd.to_datetime(dataframe.index)
    return dataframe, df


def data_pre_processing(df, lookback_size):
    """Data pre-processing - Create data for Model"""
    try:
        scaled_data = MinMaxScaler(feature_range=(0, 1))
        data_scaled_ = scaled_data.fit_transform(df)
        df.loc[:, :] = data_scaled_
        _data_ = df.to_numpy(copy=True)
        X = np.zeros(shape=(df.shape[0] - lookback_size, lookback_size, df.shape[1]))
        Y = np.zeros(shape=(df.shape[0] - lookback_size, df.shape[1]))
        timesteps = []
        for i in range(lookback_size - 1, df.shape[0] - 1):
            timesteps.append(df.index[i])
            Y[i - lookback_size + 1] = _data_[i + 1]
            for j in range(i - lookback_size + 1, i + 1):
                X[i - lookback_size + 1][lookback_size - 1 - i + j] = _data_[j]
        return X, Y, timesteps
    except Exception as e:
        print("Error while performing data pre-processing : {0}".format(e))
        return None, None, None


class DeepAnT(torch.nn.Module):
    """Model : Class for DeepAnT model"""

    def __init__(self, lookback_size, dimension):
        super(DeepAnT, self).__init__()
        self.conv1d_1_layer = torch.nn.Conv1d(in_channels=lookback_size, out_channels=16, kernel_size=1)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.conv1d_2_layer = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_1_layer = torch.nn.Linear(80, 40)
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        self.dense_2_layer = torch.nn.Linear(40, dimension)

    def forward(self, x):
        x = self.conv1d_1_layer(x)
        x = self.relu_1_layer(x)
        x = self.maxpooling_1_layer(x)
        x = self.conv1d_2_layer(x)
        x = self.relu_2_layer(x)
        x = self.maxpooling_2_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_1_layer(x)
        x = self.relu_3_layer(x)
        x = self.dropout_layer(x)
        return self.dense_2_layer(x)


class LSTMAE(torch.nn.Module):
    """Model : Class for LSTMAE model"""

    def __init__(self, dimension):
        super(LSTMAE, self).__init__()
        self.lstm_1_layer = torch.nn.LSTM(dimension, 128, 1)
        self.dropout_1_layer = torch.nn.Dropout(p=0.2)
        self.lstm_2_layer = torch.nn.LSTM(128, 64, 1)
        self.dropout_2_layer = torch.nn.Dropout(p=0.2)
        self.lstm_3_layer = torch.nn.LSTM(64, 64, 1)
        self.dropout_3_layer = torch.nn.Dropout(p=0.2)
        self.lstm_4_layer = torch.nn.LSTM(64, 128, 1)
        self.dropout_4_layer = torch.nn.Dropout(p=0.2)
        self.linear_layer = torch.nn.Linear(128, dimension)

    def forward(self, x):
        x, (_, _) = self.lstm_1_layer(x)
        x = self.dropout_1_layer(x)
        x, (_, _) = self.lstm_2_layer(x)
        x = self.dropout_2_layer(x)
        x, (_, _) = self.lstm_3_layer(x)
        x = self.dropout_3_layer(x)
        x, (_, _) = self.lstm_4_layer(x)
        x = self.dropout_4_layer(x)
        return self.linear_layer(x)


def make_train_step(model, loss_fn, optimizer):
    """Computation Return batch size data iterator"""

    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step


def train(X, Y, model_settings, anomaly_type):
    """Find Anomaly using model based computation"""
    criterion = torch.nn.MSELoss(reduction='mean')
    train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)),
                                                torch.tensor(X.astype(np.float32)))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    if model_settings['type'] == "lstmae":
        model = LSTMAE(model_settings['dimension'])
    elif model_settings['type'] == "deepant":
        model = DeepAnT(model_settings['lookback_size'], model_settings['dimension'])
    else:
        print(f"Model {model_settings['type']} is not in the set.")
        return None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    train_step = make_train_step(model, criterion, optimizer)
    progress_bar = tqdm(range(30))
    for epoch in progress_bar:
        loss_sum = 0.0
        ctr = 0
        for x_batch, y_batch in train_loader:
            loss_train = train_step(x_batch, y_batch)
            loss_sum += loss_train
            ctr += 1
        progress_bar.set_description("{0} Training Loss: {1} - Epoch: {2}".format(anomaly_type, float(loss_sum / ctr), epoch + 1))
    hypothesis = model(torch.tensor(X.astype(np.float32))).detach().numpy()
    if model_settings['type'] == "lstmae":
        loss = np.linalg.norm(hypothesis - X, axis=(1, 2))
    elif model_settings['type'] == "deepant":
        loss = np.linalg.norm(hypothesis - Y, axis=1)
    return loss.reshape(len(loss), 1)


def merge_outputs_calc_sse(totals_df, loss_df):
    totals_df['date'] = pd.to_datetime(totals_df["date"])
    totals_df['end_date'] = pd.to_datetime(totals_df["end_date"])
    analysis_df = totals_df.merge(loss_df, how='inner', left_on=['date'], right_on=['date'])
    analysis_df['sse'] = analysis_df['corr_value'].apply(lambda x: (1 - x) ** 2)
    return analysis_df


def extract_anomaly_list(threshold, start_date, analysis_df):
    anomaly_list = [1 if x[6] > float(threshold) and x[2] > pd.to_datetime(start_date) else 0 for x in
                    analysis_df.itertuples()]
    analysis_df['is_anomaly'] = anomaly_list
    anomaly_df = analysis_df.copy()
    anomaly_df.sort_values('is_anomaly', ascending=False, inplace=True)
    filter = anomaly_df['is_anomaly'] == 1
    anomaly_df.where(filter, inplace=True)
    anomaly_df = anomaly_df.dropna(how='all')
    anomaly_df.sort_values('date', ascending=True, inplace=True)
    return anomaly_df


"""# Anomaly Aggregation"""


def aggregate_anomalies(anomaly_df):
    _anomaly_df = anomaly_df.copy()
    _anomaly_df.reset_index(inplace=True, drop=True)
    _anomaly_df['date'] = pd.to_datetime(_anomaly_df['date'])
    _anomaly_df['end_date'] = pd.to_datetime(_anomaly_df['end_date'])
    indices_list = []
    start_date_index = 0
    end_date_index = 0
    for i in range(1, len(_anomaly_df)):
        duration = _anomaly_df['date'][i] - _anomaly_df['date'][start_date_index]
        if duration < timedelta(100):
            end_date_index = i
        else:
            indices_list.append((start_date_index, end_date_index, str(_anomaly_df['end_date'][end_date_index]
                                                                       - _anomaly_df['date'][start_date_index]).replace(
                ' 00:00:00', "")))
            start_date_index = i
            end_date_index = i
    indices_list.append((start_date_index, end_date_index, str(_anomaly_df['end_date'][end_date_index]
                                                               - _anomaly_df['date'][start_date_index]).replace(
        ' 00:00:00', "")))
    col_a = _anomaly_df.columns.values[2]
    col_a_name = col_a.split('_')[1]
    col_b = _anomaly_df.columns.values[3]
    col_b_name = col_b.split('_')[1]
    anomaly_df_list = []
    for item in indices_list:
        start_date = _anomaly_df['date'][item[0]]
        end_date = _anomaly_df['end_date'][item[1]]
        duration = item[2]
        avg_a = humanize.intcomma(int(np.average(_anomaly_df[col_a][item[0]:item[1] + 1])))
        avg_b = humanize.intcomma(int(np.average(_anomaly_df[col_b][item[0]:item[1] + 1])))
        min_corr = min(_anomaly_df['corr_value'][item[0]:item[1] + 1])
        max_score = max(_anomaly_df['loss'][item[0]:item[1] + 1])
        avg_sse = math.sqrt(np.sum(_anomaly_df['sse'][item[0]:item[1] + 1])) / len(
            _anomaly_df['sse'][item[0]:item[1] + 1])
        anomaly_df_list.append((start_date, end_date, duration, avg_a, avg_b, min_corr, max_score, avg_sse))
    df_anomalies = pd.DataFrame(columns=['start_date', 'end_date', 'duration', 'avg_' + col_a_name, 'avg_' + col_b_name,
                                         'min_corr', 'max_anomaly_score', 'avg_sse'])
    for data in anomaly_df_list:
        series = pd.Series(data, index=df_anomalies.columns)
        df_anomalies = df_anomalies.append(series, ignore_index=True)
    return df_anomalies


"""# Combine Anomalies"""


def transform_anomaly_output(aggregated_anomalies, anomaly_type, channel_id):
    col_headers = aggregated_anomalies.columns.values
    aggregated_anomalies['channel_id'] = channel_id
    aggregated_anomalies['duration'] = aggregated_anomalies['duration'].apply(lambda x: int(x.split(" ")[0]))
    aggregated_anomalies[col_headers[3]] = aggregated_anomalies[col_headers[3]].apply(
        lambda x: int("".join((x.split(',')))))
    aggregated_anomalies[col_headers[4]] = aggregated_anomalies[col_headers[4]].apply(
        lambda x: int("".join((x.split(',')))))
    aggregated_anomalies = aggregated_anomalies.rename(
        columns={'duration': "duration(in days)", 'avg_corr': "avg_corr(" + anomaly_type + ")", 'avg_anomaly_score':
            f"avg_anomaly_score({anomaly_type})", 'avg_sse': f"avg_sse({anomaly_type})", col_headers[3]:
                     f"{col_headers[3]}({anomaly_type})", col_headers[4]: f"{col_headers[4]}({anomaly_type})"})
    a = aggregated_anomalies.columns.values[8]
    b = aggregated_anomalies.columns.values[0]
    c = aggregated_anomalies.columns.values[1]
    d = aggregated_anomalies.columns.values[2]
    e = aggregated_anomalies.columns.values[3]
    f = aggregated_anomalies.columns.values[4]
    g = aggregated_anomalies.columns.values[5]
    h = aggregated_anomalies.columns.values[6]
    i = aggregated_anomalies.columns.values[7]
    cols = [a, b, c, d, e, f, g, h, i]
    transformed_anomalies = aggregated_anomalies[cols]
    return transformed_anomalies
