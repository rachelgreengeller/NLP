import nltk
from nltk.util import ngrams
from collections import Counter

# Download tokenizer
nltk.download('punkt')  # this will run the first time

from nltk.tokenize import word_tokenize

def read_corpus(filepath):
    with open(filepath, 'r') as f:
        text = f.read().lower()
    return word_tokenize(text)

def get_ngram_probs(tokens, n):
    n_grams = list(ngrams(tokens, n))
    n_minus_1_grams = list(ngrams(tokens, n-1))

    ngram_counts = Counter(n_grams)
    context_counts = Counter(n_minus_1_grams)

    return ngram_counts, context_counts


def calculate_probability(ngram_counts, context_counts, context, word, vocab_size, smoothing=True):
    context = tuple(context)
    ngram = context + (word,)

    if smoothing:
        numerator = ngram_counts.get(ngram, 0) + 1
        denominator = context_counts.get(context, 0) + vocab_size
    else:
        numerator = ngram_counts.get(ngram, 0)
        denominator = context_counts.get(context, 0)

        if denominator == 0:
            return 0.0  # avoid division by zero

    return numerator / denominator




# === MAIN CODE ===
tokens = read_corpus('sample.txt')
vocab = set(tokens)
vocab_size = len(vocab)

ngram_counts, context_counts = get_ngram_probs(tokens, 2)

# Example: P("nlp" | "love")
prob = calculate_probability(ngram_counts, context_counts, ["love"], "nlp", vocab_size)
print(f"P(nlp | love) = {prob:.4f}")
