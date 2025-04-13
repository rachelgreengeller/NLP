import nltk
import matplotlib.pyplot as plt
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

"""def predict_next_word(ngram_counts, context_counts, context, vocab, vocab_size):
    probs={}
    for word in vocab:
        prob=calculate_probability(ngram_counts, context_counts, context, word, vocab_size)
        probs[word]=prob

    sorted_probs = sorted(probs.items(),key=lambda x: x[1], reverse=True)
    return sorted_probs[0][0]"""

def predict_next_words(ngram_counts, context_counts, context, vocab, vocab_size,top_k=5):
    probs ={}
    for word in vocab:
        prob= calculate_probability(ngram_counts, context_counts, context, word, vocab_size)
        probs[word] = prob

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_probs


def show_prediction_chart(predictions):
    words = [w for w, _ in predictions]
    probs = [p for _, p in predictions]

    plt.figure(figsize=(8,4))
    plt.bar(word, probs, color="pink")
    plt.title('Top next Word Predictions')
    plt.xlabel('Next Word')
    plt.ylabel('Probability')
    plt.show()




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

seed= input("Type the start of the sentence: ").lower()
#ompleted = autocomplete_sentence(seed, ngram_counts, context_counts, vocab,vocab_size)
#rint(f"\nAuto-completed: {completed}")
tokens = word_tokenize(seed)
context = tokens[-1:]

top_predictions = predict_next_words(ngram_counts, context_counts, context, vocab, vocab_size)
print("\nTop predictions:")
for word, prob in top_predictions:
    print(f"{word} --> P = {prob:.4f}")

show_prediction_chart(top_predictions)

    