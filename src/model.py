import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import pickle

class NextWordPredictor:
    def __init__(self, vocab_size=5000, seq_length=5):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.model = None

    def preprocess_data(self, corpus):
        """Tokenize and create sequences from corpus."""
        self.tokenizer.fit_on_texts(corpus)
        sequences = []
        for line in corpus:
            tokens = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(tokens)):
                sequences.append(tokens[:i + 1])

        max_length = max(len(seq) for seq in sequences)
        sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

        X, y = sequences[:, :-1], sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)

        return X, y, max_length

    def build_model(self, input_length):
        """Define the RNN model."""
        model = Sequential([
            Embedding(self.vocab_size, 50, input_length=input_length),
            LSTM(100, return_sequences=True),
            LSTM(100),
            Dense(100, activation='relu'),
            Dense(self.vocab_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train(self, corpus, epochs=10, batch_size=32):
        """Train the model on the given corpus."""
        X, y, input_length = self.preprocess_data(corpus)
        self.build_model(input_length)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

        # Save tokenizer
        with open("models/tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

        self.model.save("models/rnn_model.h5")
        print("Model training complete and saved!")

if __name__ == "__main__":
    sample_corpus = [
        "This is an example sentence",
        "Machine learning helps predictive text",
        "Deep learning is useful for AI"
    ]
    predictor = NextWordPredictor()
    predictor.train(sample_corpus)
