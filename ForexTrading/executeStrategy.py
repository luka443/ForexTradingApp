from flask import Flask, render_template, request

# create the Flask application
from main import StrategiaML, Portfel

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def get_ml_data():
    pass


@app.route('/execute_strategy', methods=['POST'])
def execute_strategy():
    # get data from form
    strategy_data = request.form

    # create strategy using the data
    strategy = StrategiaML(strategy_data['nazwa'], strategy_data['opis'], strategy_data['model_ml'])

    # get data for machine learning
    dane_ml = get_ml_data()  # you will have to define this function according to your needs

    # execute strategy
    accuracy = strategy.wykonaj_strategie(Portfel(), dane_ml)

    # render results
    return render_template('results.html', accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)