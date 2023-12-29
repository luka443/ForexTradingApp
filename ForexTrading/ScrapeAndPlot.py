import os
from datetime import datetime
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from scipy import stats
import numpy as np
from Data_4_ML import Data4ML

class ScrapeAndPlot:
    def __init__(self, stock_num, instrument_name):
        self.stock_num = stock_num
        self.instrument_name = instrument_name
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.style_setup()

    def style_setup(self):
        style.use('fivethirtyeight')

    def online_price(self):
        url = f'https://finance.yahoo.com/quote/{self.stock_num}%3DX?p={self.stock_num}%3DX'
        try:
            r = requests.get(url)
            r.raise_for_status()
            content = BeautifulSoup(r.text, "html.parser")
            content = content.find('div', {"class": 'D(ib) Mend(20px)'})
            content = content.find('fin-streamer').text

            if not content:
                content = '42069'
            return content
        except requests.exceptions.RequestException as e:
            print(f"Wystąpił błąd podczas pobierania danych: {e}")
            return 'Błąd'

    def animate_price(self, i):
        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        price = self.online_price()

        col = [time_stamp, price]
        df = pd.DataFrame([col])
        df.to_csv('real_time_stock.csv', mode='a', header=False, index=False)

        data = pd.read_csv('real_time_stock.csv')
        ys = data.iloc[:, 1].values
        xs = list(range(1, len(ys) + 1))

        self.ax.clear()
        self.ax.plot(xs, ys, label=self.instrument_name)
        self.ax.set_title('live: '+self.instrument_name, fontsize=14)
        self.ax.set_xlabel('Czas')
        self.ax.set_ylabel('Cena')
        self.ax.legend()

        #plt.title('Ceny akcji (yfinance)')


    def run_animation(self):
        try:
            ani = animation.FuncAnimation(self.fig, self.animate_price, interval=1000, save_count=10)
            plt.tight_layout()

            data = yf.download(self.instrument_name+'=X', '2022-12-22', '2023-12-22')

            # Tworzenie wykresu
            plt.figure(figsize=(10, 6))
            plt.title(f'Wahania ceny 1 rok:'+ self.instrument_name+'=X')
            plt.xlabel('Data')
            plt.ylabel('Cena zamknięcia')

            # Rysowanie wykresu ceny zamknięcia
            data['Close'].plot(label='Cena Zamknięcia')

            # Dodanie linii regresji
            # Przekształcenie indeksów na liczby dla regresji
            x = range(len(data.index))
            y = data['Close'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Wyliczenie punktów linii regresji
            regression_line = [intercept + slope * i for i in x]

            # Dodanie linii regresji do wykresu
            plt.plot(data.index, regression_line, label='Srednia', color='red')

            # Dodanie legendy
            plt.legend()

            # Wyświetlenie wykresu
            plt.show()
            # plt.figure()
            # data1 = yf.download(self.stock_num+'=X', '2022-12-22', '2023-12-22')
            # plt.title('Wahania ceny 1 rok: '+ self.stock_num)
            # data1.Close.plot()
            # plt.show()
        except KeyboardInterrupt:
            print("Przerwano przez użytkownika.")

        finally:
            # Wyczyść plik CSV po zakończeniu programu
            if os.path.exists('real_time_stock.csv'):
                os.remove('real_time_stock.csv')

    def Data_ML(self):
        data = yf.download(self.instrument_name+"=X", start='2023-06-28', period='6m', interval="1h")
        datta = Data4ML(data, self.instrument_name)
        print("nigg")
        datta.Storing_Dataset()




