import nltk
from nlpkf.vectorizers import WordToVec


class DataProcessor(WordToVec):
    def __init__(self, *args, **kwargs):
        super(DataProcessor, self).__init__(*args, **kwargs)
        self.text_corpus = []
        self.vector_dim = 300  # TODO: find a way to calculate this on the fly

    def build_vocabulary(self, corpus, y=None, clean_corpus: bool = True):
        if clean_corpus:
            corpus = [self.clean_text(doc) for doc in corpus]
        super(DataProcessor, self).build_vocabulary(corpus, y=y)
        self.text_corpus = corpus
        return corpus

    def tokenize_corpus(self, corpus=None):
        data = corpus if corpus is not None else self.text_corpus
        return (self.clean_text(doc, True) for doc in data)
        return (doc.split(" ") for doc in data)

    def tokens_to_index(self, corpus):
        # Returns None when token not in vocabulary
        index_values = [[self.vocabulary[token] for token in sentence] for sentence in corpus]
        return index_values

    def to_ngrams(self, corpus, *args, **kwargs):
        if isinstance(corpus, str):
            tokens = corpus.split(" ")
        else:
            tokens = list(self.tokenize_corpus(corpus)) if isinstance(corpus[0], str) else corpus
        return [list(nltk.ngrams(doc, *args, **kwargs)) for doc in tokens]
