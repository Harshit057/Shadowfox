import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

def clean_text(text):
    """Lowercase and remove punctuation from text."""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

def generate_ngrams(text, n=2):
    """Generate n-grams from text."""
    tokens = word_tokenize(clean_text(text))
    return list(ngrams(tokens, n))

def build_ngram_model(corpus, n=2):
    """Create a frequency-based n-gram model."""
    ngram_list = []
    for sentence in corpus:
        ngram_list.extend(generate_ngrams(sentence, n))
    
    model = Counter(ngram_list)
    return model

if __name__ == "__main__":
    sample_text = ["This is a sample sentence.", "This is another test."]
    model = build_ngram_model(sample_text, n=2)
    print(model.most_common(5))  # Print top 5 n-grams
