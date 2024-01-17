from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import base64
from pathlib import Path

def download_and_strip_data(stock, period):

    data = yf.download(stock, period=period)

    DATAPATH = Path("data")
    DATANAME = f'{stock}_{period}_data.csv'
    DATA_SAVE_PATH = DATAPATH / DATANAME

    data.to_csv(DATA_SAVE_PATH)
    data = pd.read_csv(DATA_SAVE_PATH)
    data = data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    return data

#data=download_and_strip_data("GOOGL", "7d")

from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, 8):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    shifted_df_as_np = df.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    return shifted_df_as_np, scaler

#shifted_df_as_np = prepare_dataframe_for_lstm(data)

def prepering_and_spliting(shifted_df_as_np):
    #zmiana do numpy

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    X = dc(np.flip(X, axis=1))

    #dzielenie datasetu w stosunku 95:5
    split_index = int(len(X) * 0.95)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]


    X_train = X_train.reshape((-1, 7, 1))
    X_test = X_test.reshape((-1, 7, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))


    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return X_train, y_train, X_test, y_test

#X_train, y_train, X_test, y_test = prepering_and_spliting(shifted_df)

def prepering_and_spliting2(shifted_df_as_np):
    #zmiana do numpy
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    X = dc(np.flip(X, axis=1))

    X = X.reshape((-1, 7, 1))
    y = y.reshape((-1, 1))


    X = torch.tensor(X).float()
    y = torch.tensor(y).float()


    return X, y

def prepering_for_prediction(shifted_df_as_np):
    #zmiana do numpy
    X = shifted_df_as_np[:, 0:7]

    X = dc(np.flip(X, axis=1))
    X = X.reshape((-1, 7, 1))

    X = torch.tensor(X).float()

    return X
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_loaders(X_train, y_train, X_test, y_test):

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0], batch[1]
        break

    return train_loader, test_loader

#train_loader, test_loader = get_loaders(X_train, y_train, X_test, y_test)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


#model = LSTM(1, 4, 1)
def train_one_epoch(model, train_loader, optimizer, loss_function, epoch):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0], batch[1]

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch(model, test_loader, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0], batch[1]

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)


def train_and_validate(model, train_loader, test_loader):
    learning_rate = 0.001
    num_epochs = 10
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, loss_function, epoch)
        validate_one_epoch(model, test_loader, loss_function)

#train_and_validate(model, train_loader, test_loader)
def save_model(model, name):
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
    MODEL_NAME = f"{name}_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # 3. Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)

def load_model(ticker):
    MODEL_PATH = Path("models")
    MODEL_NAME = f"{ticker}_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    loaded_model = LSTM(1, 4, 1)
    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    return loaded_model

#ZAPREZENTOWANIE CO SIE MODEL NAUCZYŁ
# with torch.no_grad():
#     predicted = model(X_train).to('cpu').numpy()
#
# plt.plot(y_train, label='Actual Close')
# plt.plot(predicted, label='Predicted Close')
# plt.xlabel('Day')
# plt.ylabel('Close')
# plt.legend()
# plt.show()
#
# train_predictions = predicted.flatten()
#
# dummies = np.zeros((X_train.shape[0], 8))
# dummies[:, 0] = train_predictions
# dummies = scaler.inverse_transform(dummies)
#
# train_predictions = dc(dummies[:, 0])
#
# dummies = np.zeros((X_train.shape[0], 8))
# dummies[:, 0] = y_train.flatten()
# dummies = scaler.inverse_transform(dummies)
#
# new_y_train = dc(dummies[:, 0])
#
#
# plt.plot(new_y_train, label='Actual Close')
# plt.plot(train_predictions, label='Predicted Close')
# plt.xlabel('Day')
# plt.ylabel('Close')
# plt.legend()
# plt.show()

def predict_and_plot(model, X, y, scaler):

    test_predictions = model(X).detach().cpu().numpy().flatten()

    dummies = np.zeros((X.shape[0], 8))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])

    dummies = np.zeros((X.shape[0], 8))
    dummies[:, 0] = y.flatten()
    dummies = scaler.inverse_transform(dummies)

    new_y = dc(dummies[:, 0])


    plt.style.use('dark_background')
    plt.figure(figsize=(14, 7))
    plt.grid(True)
    plt.plot(new_y, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.xlabel('Day')

    plt.ylabel('Close')
    plt.legend()
    # plt.show()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/ class='img-fluid rounded'>"

def predict(model, X, scaler):

    test_prediction = model(X).detach().cpu().numpy().flatten()

    dummies = np.zeros((X.shape[0], 8))
    dummies[:, 0] = test_prediction
    dummies = scaler.inverse_transform(dummies)

    test_prediction = dc(dummies[:, 0])

    return test_prediction



# data = download_and_strip_data("GOOGL", "20y")
# shifted_df_as_np, scaler = prepare_dataframe_for_lstm(data)
# X_train, y_train, X_test, y_test = prepering_and_spliting(shifted_df_as_np)
# train_loader, test_loader = get_loaders(X_train, y_train, X_test, y_test)
# model = LSTM(1, 4, 1)
# train_and_validate(model, train_loader, test_loader)
# predict_and_plot(model, X_test, y_test, scaler)
#
#
# data2 = download_and_strip_data("GOOGL", "8d")
# shifted_df_as_np, scaler = prepare_dataframe_for_lstm(data2)
# X, y = prepering_and_spliting2(shifted_df_as_np)
# predict_and_plot(model, X, y, scaler)
# X=prepering_for_prediction(shifted_df_as_np)
# predict(model, X, scaler)
