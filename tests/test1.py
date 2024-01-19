import unittest
import pandas as pd

import os

from model_trainer import download_and_strip_data


class TestDownloadAndStripData(unittest.TestCase):

    def test_download_and_strip_data(self):
        # Wywołanie funkcji do testowania z przykładowymi danymi
        stock = "AAPL"
        period = "1d"
        result = download_and_strip_data(stock, period)

        # Sprawdzanie, czy wynik jest DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Sprawdzanie, czy DataFrame zawiera odpowiednie kolumny
        expected_columns = ['Date', 'Close']
        self.assertListEqual(list(result.columns), expected_columns)

        # Sprawdzanie, czy plik został utworzony
        file_path = f"data/{stock}_{period}_data.csv"
        self.assertTrue(os.path.exists(file_path))

        # Usuwanie utworzonego pliku po teście
        os.remove(file_path)

if __name__ == '__main__':
    unittest.main()
