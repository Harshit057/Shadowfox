import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import words
import re

nltk.download("words")

class RNNPredictor:
    def __init__(self):
        self.word_set = set(words.words())  # English vocabulary
        self.model = self.load_model()

    def load_model(self):
        """Load trained RNN model."""
        try:
            return tf.keras.models.load_model("models/rnn_model.h5")
        except:
            print("Model not found! Train and save the model first.")
            return None

    def clean_text(self, text):
        """Basic text preprocessing."""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.strip()

    def autocorrect(self, word):
        """Simple autocorrect function using known words."""
        if word in self.word_set:
            return word  # Word is correct

        # Find the closest word using edit distance
        closest_word = min(self.word_set, key=lambda w: nltk.edit_distance(word, w))
        return closest_word

    def predict_next_word(self, text):
        """Predict the next word using RNN and provide top 3 suggestions."""
        text = self.clean_text(text)
        words = text.split()

        # Autocorrect the last word
        if words:
            words[-1] = self.autocorrect(words[-1])

        corrected_text = " ".join(words)

        # If RNN model isn't available or text is too short
        if len(words) < 2 or self.model is None:
            return corrected_text, ["..."]

        # Convert words to input format for RNN
        input_data = np.array(words[-2:]).reshape(1, -1)
        predictions = self.model.predict(input_data)

        # Get the top 3 predicted words
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        predicted_words = [self.get_word_from_index(idx) for idx in top_indices]

        return corrected_text, predicted_words

    def get_word_from_index(self, index):
        """Convert index to actual word (Dummy implementation)."""
        word_list = ["hello", "world", "great", "python", "AI", "future"]
        return word_list[index % len(word_list)]
