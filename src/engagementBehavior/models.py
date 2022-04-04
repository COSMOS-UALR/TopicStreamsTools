import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm



def data_pre_processing(df, lookback_size):
    """Data pre-processing - Create data for Model"""
    df = df.fillna(df.mean(numeric_only=True))
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
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
