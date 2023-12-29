from tkinter import Tk, Label, Entry, Button, W, E, N, S
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import graphviz
from ScrapeAndPlot import ScrapeAndPlot


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
        scrape_and_plot.Data_ML()
        self.root.withdraw()

        scrape_and_plot.run_animation()


        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = MainApp(root)
    root.mainloop()