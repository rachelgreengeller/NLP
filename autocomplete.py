import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

from nltk.tokenize import word_tokenize

def read_corpus(filepath):
    with open(filepath,'r') as f:
        text=f.read().lower()
    return word_tokenize(text)

def get_ngrams_probs(token,n):
    n_grams=list(ngrams(token,n))
    n_minus_1_grams=list(ngrams(tokens,n-1))

    ngram_counts=Counter(n_grams)
    context_counts=Counter(n_minus_1_grams)

    return ngram_counts, context_counts

def calculate_probability(ngram_counts, context_counts, context, word, vocab_size):
    context=tuple(context)
    ngram= context + (word,)

    numerator = ngram_counts.get(ngram,0)+1
    denominator = context_counts.get(context,0) + vocab_size

    return numerator / denominator 

def predict_next_word(ngram_counts, context_counts, context, vocab, vocab_size):
    probs={}
    for word in vocab:
        prob=calculate_probability(ngram_counts, context_counts, context, word, vocab_size)
        probs[word]=prob

    sorted_probs = sorted(probs.items(),key=lambda x: x[1], reverse=True)
    return sorted_probs[0][0]

def autocomplete_sentence(seed_text, ngram_counts, context_counts, vocab, vocab_size, steps=5):
    tokens=word_tokenize(seed_text.lower())
    for _ in range(steps):
        context=tokens[-1:]
        next_word=predict_next_word(ngram_counts,context_counts,context,vocab, vocab_size)
        tokens.append(next_word)
    return ' '.join(tokens)

tokens = read_corpus('sample.txt')
vocab = set(tokens)
vocab_size = len(vocab)

ngram_counts, context_counts = get_ngrams_probs(tokens,2)

seed= input("Type the start of the sentence: ")
completed = autocomplete_sentence(seed, ngram_counts, context_counts, vocab,vocab_size)
print(f"\nAuto-completed: {completed}")

    