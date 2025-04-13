import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')

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
            return 0.0

    return numerator / denominator

def suggest_next_words(ngram_counts, context_counts, context, vocab, vocab_size, top_n=3):
    probs = {}
    for word in vocab:
        prob = calculate_probability(ngram_counts, context_counts, context, word, vocab_size)
        probs[word] = prob

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[:top_n]

# === MAIN ===
tokens = read_corpus('sample.txt')
vocab = set(tokens)
vocab_size = len(vocab)

ngram_counts, context_counts = get_ngram_probs(tokens, 2)

# Input word
input_word = input("Enter a word: ").lower()
suggestions = suggest_next_words(ngram_counts, context_counts, [input_word], vocab, vocab_size)

print(f"\nTop suggestions after '{input_word}':")
for word, prob in suggestions:
    print(f"{word}  -->  P = {prob:.4f}")
