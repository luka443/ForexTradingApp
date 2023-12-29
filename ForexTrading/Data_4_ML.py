import os
from datetime import datetime
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier


class Data4ML:
    def __init__(self, data, inst_name):
        self.data = data
        self.inst_name = inst_name

    def Storing_Dataset(self):
        some_stock = self.data
        some_stock["Prediction_NextInterval"] = some_stock["Close"].shift(-1)
        # tworze sobie tabele ktora ma cene z poprzedniego interwału np dna 30 minut itp TERAZ JEST interval=1h

        some_stock["Target"] = (some_stock["Prediction_NextInterval"] > some_stock["Close"]).astype(int)
        # tworze kolumne target jak cena jest z kolejnegointerwału jest wyzsza to 1 jak nie to 0(1 jak cena wzrosła) w pliku csv mozna sparawdzic

        print(self.data)
        some_stock.to_csv('Data' + self.inst_name + '.csv', )  # osobno pliki dla kazdego kursuu jaki chce uzytkownik

    def TrainingModel(self):
        model = RandomForestClassifier(n_estimators=100, min_samples_split=100,
                                       random_state=1)  # im wyzsze n_estimators to tym dokladniej i random_state=1 czyli jaksie odpali kilka razy to da to samo
        prediction_used = ["Close", "Open", "High", "Low"]
        train = (self.data).iloc[:-250]
        test = (self.data).iloc[-250:]
        model.fit(train[prediction_used],
                  train["Target"])

        # TODO
