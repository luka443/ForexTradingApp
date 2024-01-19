import unittest
from wsgi import train_model, LSTM

class TestTrainModel(unittest.TestCase):

    def test_train_model_returns_LSTM_object(self):
        # Wybór tickera do testowania
        ticker = 'AAPL'

        # Wywołanie funkcji train_model
        model = train_model(ticker)

        # Sprawdzenie, czy zwrócony obiekt jest instancją klasy LSTM
        self.assertIsInstance(model, LSTM)

if __name__ == '__main__':
    unittest.main()