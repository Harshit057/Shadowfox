from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from predict import RNNPredictor

# Full QWERTY Layout
QWERTY_KEYS = [
    "qwertyuiop", "asdfghjkl", "zxcvbnm"
]

class AIKeyboardApp(App):
    def build(self):
        self.predictor = RNNPredictor()

        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Input Box
        self.text_input = TextInput(font_size=20, size_hint=(1, 0.2), multiline=False)
        self.text_input.bind(text=self.update_prediction)
        layout.add_widget(self.text_input)

        # Autocorrected Text Display
        self.corrected_label = Label(text="Autocorrected: ", font_size=18, color=(0, 1, 0, 1))
        layout.add_widget(self.corrected_label)

        # Prediction Buttons (Dynamically Updated)
        self.prediction_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.prediction_buttons = [Button(text="", font_size=18, on_press=self.select_prediction) for _ in range(3)]
        for btn in self.prediction_buttons:
            self.prediction_layout.add_widget(btn)
        layout.add_widget(self.prediction_layout)

        # Keyboard Layout
        for row in QWERTY_KEYS:
            row_layout = BoxLayout(orientation='horizontal', spacing=5)
            for letter in row:
                btn = Button(text=letter, font_size=24)
                btn.bind(on_press=self.add_letter)
                row_layout.add_widget(btn)
            layout.add_widget(row_layout)

        # Space and Backspace
        bottom_layout = BoxLayout(orientation='horizontal', spacing=5)
        space_btn = Button(text="Space", font_size=20, on_press=self.add_space)
        backspace_btn = Button(text="⌫", font_size=20, on_press=self.backspace)
        bottom_layout.add_widget(space_btn)
        bottom_layout.add_widget(backspace_btn)
        layout.add_widget(bottom_layout)

        return layout

    def add_letter(self, instance):
        """Add the pressed letter to the text input."""
        self.text_input.text += instance.text

    def add_space(self, instance):
        """Add a space to the text input."""
        self.text_input.text += " "

    def backspace(self, instance):
        """Remove the last character from the text input."""
        self.text_input.text = self.text_input.text[:-1]

    def update_prediction(self, instance, value):
        """Autocorrect and update next-word suggestions."""
        if value.strip():
            corrected_text, predictions = self.predictor.predict_next_word(value)
            self.corrected_label.text = f"Autocorrected: {corrected_text}"
            for i, word in enumerate(predictions[:3]):
                self.prediction_buttons[i].text = word
        else:
            self.corrected_label.text = "Autocorrected: "
            for btn in self.prediction_buttons:
                btn.text = ""

    def select_prediction(self, instance):
        """Insert the selected prediction into the text input."""
        self.text_input.text += " " + instance.text

if __name__ == "__main__":
    AIKeyboardApp().run()
