import tkinter as tk
from predict import RNNPredictor

class AutocorrectKeyboard:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Keyboard")
        self.root.geometry("500x300")

        self.predictor = RNNPredictor()

        # Label
        self.label = tk.Label(root, text="Type a sentence:", font=("Arial", 14))
        self.label.pack(pady=10)

        # Text Entry
        self.text_entry = tk.Entry(root, font=("Arial", 14), width=40)
        self.text_entry.pack(pady=10)
        self.text_entry.bind("<KeyRelease>", self.update_prediction)

        # Prediction Label
        self.suggestion_label = tk.Label(root, text="Prediction: ", font=("Arial", 14), fg="blue")
        self.suggestion_label.pack(pady=10)

    def update_prediction(self, event):
        """Predict and update the next word suggestion."""
        text = self.text_entry.get()
        if text.strip():
            prediction = self.predictor.predict_next_word(text)
            self.suggestion_label.config(text=f"Prediction: {prediction}")
        else:
            self.suggestion_label.config(text="Prediction: ")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutocorrectKeyboard(root)
    root.mainloop()
