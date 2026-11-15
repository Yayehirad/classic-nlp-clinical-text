"""Bag-of-words Naive Bayes helpers built on NLTK, plus evaluation utilities."""

import math

from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

from .text_prep import PUNCTUATION

_STRIP = str.maketrans("", "", PUNCTUATION)


def preprocess(text):
    """Lowercase, strip punctuation and tokenise one document."""
    return word_tokenize(str(text).lower().translate(_STRIP))


def build_bow(tokens):
    """Word-frequency dictionary for a token list, e.g. ['a', 'b', 'a'] -> {'a': 2, 'b': 1}."""
    bow = {}
    for token in tokens:
        bow[token] = bow.get(token, 0) + 1
    return bow


def bow_to_features(bow):
    """Binary presence features in the {word: True} format NLTK expects."""
    return {word: True for word in bow}


def build_features(train_texts, train_labels, test_texts, test_labels):
    """Tokenise documents and build NLTK feature sets with a closed vocabulary.

    The vocabulary comes from the training documents only; unseen test words
    are dropped. Returns (train_features, test_features, vocabulary).
    """
    train_tokens = [preprocess(t) for t in train_texts]
    vocabulary = set().union(*train_tokens)
    train_features = [(bow_to_features(build_bow(tok)), y)
                      for tok, y in zip(train_tokens, train_labels)]
    test_features = []
    for text, y in zip(test_texts, test_labels):
        tokens = [t for t in preprocess(text) if t in vocabulary]
        test_features.append((bow_to_features(build_bow(tokens)), y))
    return train_features, test_features, vocabulary


def train_naive_bayes(train_features):
    """Train NLTK's Naive Bayes classifier."""
    return NaiveBayesClassifier.train(train_features)


def predict(classifier, features):
    """Predicted labels for a list of (featureset, label) pairs."""
    return [classifier.classify(f) for f, _ in features]


def top_words_per_class(classifier, n=10, min_class_prob=0.05):
    """Words most characteristic of each class.

    For each class c and word w, the score is log P(w present | c) minus the mean
    log P(w present | other classes). Words present in fewer than `min_class_prob`
    of class-c documents are skipped so rare one-off words don't dominate.
    """
    labels = classifier.labels()
    words = {fname for (_, fname) in classifier._feature_probdist}
    probs = {(c, w): classifier._feature_probdist[(c, w)].prob(True)
             for c in labels for w in words}
    result = {}
    for c in labels:
        scored = []
        for w in words:
            p_c = probs[(c, w)]
            if p_c < min_class_prob:
                continue
            others = [math.log(probs[(o, w)]) for o in labels if o != c]
            scored.append((math.log(p_c) - sum(others) / len(others), w))
        result[c] = [w for _, w in sorted(scored, reverse=True)[:n]]
    return result
