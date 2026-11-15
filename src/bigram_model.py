"""A bigram language model with add-k (Laplace) smoothing, built from plain counts."""

import math
import random

from nltk.tokenize import sent_tokenize, word_tokenize

START, END = "<s>", "<e>"


def tokenize_corpus(processed_text):
    """Split preprocessed text into sentences, each a list of word tokens.

    Adapted from a university NLP lab exercise.
    """
    return [word_tokenize(sentence) for sentence in sent_tokenize(processed_text)]


def count_n_grams(data, n, start_token=START, end_token=END):
    """Count n-grams across tokenised sentences.

    Each sentence is padded with n start tokens and one end token.
    Returns a dict mapping n-gram tuples to counts.

    Adapted from a university NLP lab exercise.
    """
    n_grams = {}
    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]
        for i in range(len(sentence) - n + 1):
            n_gram = tuple(sentence[i:i + n])
            n_grams[n_gram] = n_grams.get(n_gram, 0) + 1
    return n_grams


def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts,
                         vocabulary_size, k=1.0):
    """P(word | previous n-gram) with add-k smoothing (k=1 is Laplace).

    Adapted from a university NLP lab exercise.
    """
    previous_n_gram = tuple(previous_n_gram)
    denominator = n_gram_counts.get(previous_n_gram, 0) + k * vocabulary_size
    numerator = n_plus1_gram_counts.get(previous_n_gram + (word,), 0) + k
    return numerator / denominator


def sentence_log_probability(sentence, n_gram_counts, n_plus1_gram_counts,
                             vocabulary_size, k=1.0):
    """Natural-log probability of a tokenised sentence, including the end token.

    Summing logs avoids the underflow you get from multiplying many small numbers.
    Adapted from a university NLP lab exercise.
    """
    n = len(next(iter(n_gram_counts)))
    sentence = tuple([START] * n + sentence + [END])
    log_sum = 0.0
    for t in range(n, len(sentence)):
        log_sum += math.log(estimate_probability(
            sentence[t], sentence[t - n:t], n_gram_counts, n_plus1_gram_counts,
            vocabulary_size, k))
    return log_sum


class BigramModel:
    """Bigram language model trained on a list of tokenised sentences."""

    def __init__(self, sentences, k=1.0):
        self.k = k
        self.unigram_counts = count_n_grams(sentences, 1)
        self.bigram_counts = count_n_grams(sentences, 2)
        self.vocab = {w for (w,) in self.unigram_counts}
        self.n_train_tokens = sum(len(s) for s in sentences)

    def log_prob(self, tokens):
        """Log-probability of one tokenised sentence."""
        return sentence_log_probability(tokens, self.unigram_counts, self.bigram_counts,
                                        len(self.vocab), self.k)

    def perplexity(self, sentences):
        """exp(-average log-probability per predicted token), end tokens included."""
        total_log_prob, n_tokens = 0.0, 0
        for tokens in sentences:
            total_log_prob += self.log_prob(tokens)
            n_tokens += len(tokens) + 1
        return math.exp(-total_log_prob / n_tokens)

    def generate(self, max_words=25, seed=None):
        """Sample a sentence word by word from the (unsmoothed) bigram counts."""
        rng = random.Random(seed)
        if not hasattr(self, "_next_words"):
            self._next_words = {}
            for (w1, w2), c in self.bigram_counts.items():
                self._next_words.setdefault(w1, []).append((w2, c))
        word, out = START, []
        while len(out) < max_words:
            candidates = [(w, c) for w, c in self._next_words.get(word, []) if w != START]
            if not candidates:
                break
            words, weights = zip(*candidates)
            word = rng.choices(words, weights=weights)[0]
            if word == END:
                break
            out.append(word)
        return " ".join(out)


def predict_author(sentence, models):
    """Return the author whose model gives the sentence the highest log-probability.

    `models` maps an author name to a BigramModel. Also returns the scores.
    """
    tokens = word_tokenize(sentence.lower())
    scores = {author: model.log_prob(tokens) for author, model in models.items()}
    return max(scores, key=scores.get), scores
