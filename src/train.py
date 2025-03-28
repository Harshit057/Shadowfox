import pickle
from preprocess import build_ngram_model

def train_ngram_model(corpus, n=2, save_path="models/ngram_model.pkl"):
    """Train and save an n-gram model."""
    model = build_ngram_model(corpus, n)
    
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved at {save_path}")

if __name__ == "__main__":
    sample_corpus = [
        "This is an example sentence.",
        "Machine learning helps predictive text."
    ]
    train_ngram_model(sample_corpus, n=2)
