import unittest
import os

from model_trainer import LSTM, save_model


class TestSaveModel(unittest.TestCase):

    def test_save_model_creates_file(self):
        # Tworzenie przykładowego modelu
        test_model = LSTM(1, 4, 1)
        model_name = "test_model"

        # Wywołanie funkcji zapisującej model
        save_model(test_model, model_name)

        # Ścieżka, gdzie powinien zostać zapisany model
        expected_file_path = f"models/{model_name}_model.pth"

        # Sprawdzanie, czy plik został utworzony
        self.assertTrue(os.path.isfile(expected_file_path))

        # Usuwanie pliku po teście
        os.remove(expected_file_path)

if __name__ == '__main__':
    unittest.main()