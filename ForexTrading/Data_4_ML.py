import os
from datetime import datetime
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt


class Data4ML:
    def __init__(self, data, inst_name):
        self.data = data
        self.inst_name = inst_name

    def Storing_Dataset(self):
        some_stock = self.data
        some_stock["Prediction_NextInterval"] = some_stock["Close"].shift(-1)
        some_stock["Target"] = (some_stock["Prediction_NextInterval"] > some_stock["Close"]).astype(int)
        print(self.data)
        some_stock.to_csv('Data' + self.inst_name + '.csv')

    def TrainingModel(self):
        model = RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1)
        prediction_used = ["Close", "Open", "High", "Low"]
        train = (self.data).iloc[:-100]
        test = (self.data).iloc[-100:]
        model.fit(train[prediction_used], train["Target"])
        RandomForestClassifier(min_samples_split=100, random_state=1)
        preds = model.predict(test[prediction_used])
        preds = pd.Series(preds, index=test.index)
        precision_score(test["Target"], preds)
        combined = pd.concat([test["Target"], preds], axis=1)

        def predict(train, test, predictors, model):
            model.fit(train[predictors], train["Target"])
            preds2 = model.predict_proba(test[predictors])[:, 1]
            preds2[preds2 >= .5] = 1
            preds2[preds2 < .5] = 0
            preds2 = pd.Series(preds2, index=test.index, name="Predictions")
            combined2 = pd.concat([test["Target"], preds2], axis=1)
            return combined2

        def backtest(data, model, predictors, start=2500, step=250):
            all_predictions = []

            for i in range(start, data.shape[0], step):
                train2 = data.iloc[0:i].copy()
                test2 = data.iloc[i:(i + step)].copy()
                predictions2 = predict(train2, test2, predictors, model)
                all_predictions.append(predictions2)

            return pd.concat(all_predictions)

        horizons = [2, 5, 60, 250]
        new_predictors = []

        for horizon in horizons:
            rolling_averages = self.data.rolling(horizon).mean()
            ratio_column = f"Close_Ratio_{horizon}"
            self.data[ratio_column] = self.data["Close"] / rolling_averages["Close"]
            trend_column = f"Trend_{horizon}"
            self.data[trend_column] = self.data.shift(1).rolling(horizon).sum()["Target"]
            new_predictors += [ratio_column, trend_column]

        self.data = self.data.dropna(subset=(self.data).columns[(self.data).columns != "Prediction_NextInterval"])

        predictions = backtest(self.data, model, new_predictors)

        predictions["Predictions"].value_counts()
        print("Prawdopodobienstwo dobrej decyzji modelu:",
              precision_score(predictions["Target"], predictions["Predictions"]))
        print("\n")
        print("Prawdopodobienstwo, że cena poszła w dół (0), że poszła w górę(1)")
        print(predictions["Target"].value_counts() / predictions.shape[0])


