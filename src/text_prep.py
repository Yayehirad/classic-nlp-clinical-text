"""Text cleaning shared by the bigram model and the Naive Bayes classifier."""

import re
import string

from nltk.tokenize import sent_tokenize

# Punctuation to strip: the standard set plus curly quotes and dashes, but keeping
# the full stop so sentence boundaries survive. Both original pipelines used this set.
PUNCTUATION = (string.punctuation + "“”-’‘—").replace(".", "")


def preprocess_text(text):
    """Lowercase, strip punctuation (except full stops) and normalise whitespace.

    Returns one string with each full stop surrounded by spaces, ready for
    sentence and word tokenisation. Volume/chapter headings from the Gutenberg
    novels are removed.

    Adapted from starter code supplied with the original university coursework.
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    chars = "".join(ch.lower() for ch in text if ch not in PUNCTUATION)
    sents = ""
    for sent in sent_tokenize(chars):
        sents += sent.replace(".", " . ")
    return re.sub(r"volume [mdclxvi]+ chapter [mdclxvi]+ ", "", sents)
