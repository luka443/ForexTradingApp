import os.path
from model_trainer import *
from flask import Flask, render_template, request
import yfinance as yf

app = Flask(__name__)

def train_model(ticker):
    data = download_and_strip_data(ticker, "20y")
    shifted_df_as_np, scaler = prepare_dataframe_for_lstm(data)
    X_train, y_train, X_test, y_test = prepering_and_spliting(shifted_df_as_np)
    train_loader, test_loader = get_loaders(X_train, y_train, X_test, y_test)
    model = LSTM(1, 4, 1)
    train_and_validate(model, train_loader, test_loader)
    save_model(model, ticker)
    return model


def check_and_return_model(ticker):
    MODEL_PATH = Path(f"models/{ticker}_model.pth")
    if os.path.exists(MODEL_PATH):
        model = load_model(ticker)
    else:
        model = train_model(ticker)

    return model


def get_predicted_plots(model, ticker, period):
    data2 = download_and_strip_data(ticker, period)
    shifted_df_as_np, scaler = prepare_dataframe_for_lstm(data2)
    X, y = prepering_and_spliting2(shifted_df_as_np)
    plot = predict_and_plot(model, X, y, scaler)
    return plot #string img


def get_prediction(model, ticker):
    data = download_and_strip_data(ticker, "8d")
    shifted_df_as_np, scaler = prepare_dataframe_for_lstm(data)
    X = prepering_for_prediction(shifted_df_as_np)
    prediction = predict(model, X, scaler)

    return prediction[0]

def get_current_price(ticker):
    ticker = yf.Ticker(ticker)
    # Get stock info
    stock_info = ticker.info
    return stock_info['currentPrice']

def plot_stock_price(ticker_symbol, period):
    # Pobranie danych o cenie akcji
    ticker = yf.Ticker(ticker_symbol)
    ticker_df = ticker.history(period=period)

    plt.style.use('dark_background')
    # Stworzenie wykresu
    plt.figure(figsize=(14, 7))
    plt.plot(ticker_df['Close'])
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/ class='img-fluid rounded'>"



popular_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "AMD", "BA", "PYPL", "MS"]

def get_stock_info(tickers):
    # Pusta ramka danych do przechowywania informacji
    stocks_df = [["Symbol", "Company Name", "Current Price", "Change $", "Change %", "Volume"]]

    for ticker in tickers:
        stock_info = yf.Ticker(ticker)
        info_dict = stock_info.info
        #print(info_dict)
        current_price = info_dict["currentPrice"]
        previous_close = info_dict["previousClose"]
        volume = info_dict["volume"]

        change_dollar = current_price - previous_close
        change_percent = (change_dollar / previous_close) * 100

        stocks_df.append([
            ticker,
            info_dict['shortName'],
            current_price,
            change_dollar,
            change_percent,
            volume])

    return stocks_df


@app.route('/', methods=['POST', 'GET'])
def index():
    ticker = request.form.get('ticker')
    if ticker is None:
        ticker = 'GOOGL'
    tbl = get_stock_info(popular_tickers)
    plot1 = plot_stock_price(ticker, '1mo')
    plot2 = plot_stock_price(ticker, '1y')

    model = check_and_return_model(ticker)
    plot_ml1 = get_predicted_plots(model, ticker, '1mo')
    plot_ml2 = get_predicted_plots(model, ticker, '20y')
    plot_ml3 = get_predicted_plots(model, ticker, '10d')
    prediction_ml= get_prediction(model, ticker)
    current_price = get_current_price(ticker)
    change_dollar = prediction_ml-current_price
    change_percent = (change_dollar / prediction_ml) * 100

    return render_template("main.html",
                           table= tbl,
                           plot1=plot1,
                           plot2=plot2,
                           plot_ml1=plot_ml1,
                           plot_ml2=plot_ml2,
                           plot_ml3=plot_ml3,
                           prediction=round(prediction_ml,4),
                           change=round(change_dollar,4),
                           change2=round(change_percent,4),
                           ticker=ticker)


