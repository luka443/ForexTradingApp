from tkinter import Tk, Label, Entry, Button, W, E, N, S
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import graphviz
from ScrapeAndPlot import ScrapeAndPlot



# to bylo potrzebne dowygenerowania umla bo mi sie rysowac nei chcialo xd ale moze tzreba bedzie cos z tego wykorzystac przez to
# class Transakcja:
#     def __init__(self, instrument, ilosc, cena, data):
#         self.instrument = instrument
#         self.ilosc = ilosc
#         self.cena = cena
#         self.data = data
#
#     def __str__(self):
#         return f"Transakcja: {self.instrument.symbol} - {self.ilosc} jednostek, Cena: {self.cena}, Data: {self.data}"
#
#
# class Portfel:
#     def __init__(self):
#         self.transakcje = []
#
#     def dodaj_transakcje(self, transakcja):
#         self.transakcje.append(transakcja)
#
#     def __str__(self):
#         if not self.transakcje:
#             return "Pusty portfel"
#         else:
#             transakcje_str = "\n".join(str(transakcja) for transakcja in self.transakcje)
#             return f"Portfel:\n{transakcje_str}"
#
#
# class Strategia:
#     def __init__(self, nazwa, opis):
#         self.nazwa = nazwa
#         self.opis = opis
#
#     def wykonaj_strategie(self, portfel):
#         pass
#
#     def __str__(self):
#         return f"Strategia: {self.nazwa}\nOpis: {self.opis}"
#
#
# class StrategiaML(Strategia):
#     def __init__(self, nazwa, opis, model_ml):
#         super().__init__(nazwa, opis)
#         self.model_ml = model_ml
#
#     def przygotuj_dane(self, dane_ml):
#         # Here is a placeholder for your actual method
#         dane_uczenia, etykiety = dane_ml, None
#         return dane_uczenia, etykiety
#
#     def zastosuj_strategie_na_podstawie_prognoz(self, portfel, prognozy):
#         # Here is a placeholder for your actual method
#         pass
#
#     def wykonaj_strategie(self, portfel, dane_ml):
#         # Przygotuj dane
#         dane_uczenia, etykiety = self.przygotuj_dane(dane_ml)
#
#         # Podziel dane na zbiór treningowy i testowy
#         dane_treningowe, dane_testowe, etykiety_treningowe, etykiety_testowe = train_test_split(
#             dane_uczenia, etykiety, test_size=0.2, random_state=42
#         )
#
#         # Wytrenuj model ML
#         self.model_ml.fit(dane_treningowe, etykiety_treningowe)
#
#         # Dokonaj prognoz na zbiorze testowym
#         prognozy = self.model_ml.predict(dane_testowe)
#
#         # Ocen skuteczność modelu
#         dokladnosc = accuracy_score(etykiety_testowe, prognozy)
#
#         # Zastosuj strategię handlową w oparciu o prognozy
#         self.zastosuj_strategie_na_podstawie_prognoz(portfel, prognozy)
#
#         return dokladnosc
#
#
# class StrategiaKupSrednia(Strategia):
#     def __init__(self, nazwa, opis, okres_sredniej):
#         super().__init__(nazwa, opis)
#         self.okres_sredniej = okres_sredniej
#
#     def wykonaj_strategie(self, portfel):
#         pass
#

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Scrape and Plot")

        # Ustawienia rozmiaru i umiejscowienia okna
        width = 300
        height = 150
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Ustawienia, aby nie można było zmieniać rozmiaru okna
        self.root.resizable(width=False, height=False)

        self.instrument_label = Label(root, text="Nazwa instrumentu(wpisz np:EURUSD albo AUDUSD)")
        self.instrument_entry = Entry(root, width=30)
        self.run_button = Button(root, text="Uruchom", command=self.run_scrape_and_plot)

        # Ustawienia do rozmieszczenia widgetów w siatce
        self.instrument_label.grid(row=0, column=0, padx=10, pady=10, sticky=E)
        self.instrument_entry.grid(row=1, column=0, padx=20, pady=20, sticky=W)
        self.run_button.grid(row=2, columnspan=2, pady=10)
        self.instrument_entry.bind("<Return>", lambda event: self.run_scrape_and_plot())

    def run_scrape_and_plot(self):
        instrument_name = self.instrument_entry.get()
        scrape_and_plot = ScrapeAndPlot(instrument_name, instrument_name)

        self.root.withdraw()

        scrape_and_plot.run_animation()

        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = MainApp(root)
    root.mainloop()