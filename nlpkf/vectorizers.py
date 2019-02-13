import os
import numpy as np
from typing import Callable
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from nlpkf.tokenizer import Tokenizer, clean_text


def create_word_vectorizer(tokenizer_kwargs, return_tokenizer: bool = False, *args, **kwargs):
    tokenizer = Tokenizer(**tokenizer_kwargs)
    countvec = CountVectorizer(tokenizer=tokenizer.tokenize_text, *args, **kwargs)
    return countvec, tokenizer if return_tokenizer else countvec


def get_tokenized_doc(text, nlp=None, *args, **kwargs):

    if nlp is None:
        clean, tok = clean_text(
            text, return_tokens=False, return_tokenizer=True, nlp=nlp, *args, **kwargs
        )
        nlp = tok.nlp
    else:
        clean = clean_text(
            text, return_tokens=False, return_tokenizer=False, nlp=nlp, *args, **kwargs
        )
    doc = nlp(clean)
    return doc


def text_to_word_vecs(text, *args, **kwargs):
    doc = get_tokenized_doc(text=text, *args, **kwargs)
    return [sent.vector for sent in doc]


def text_to_nchunk_vecs(text, *args, **kwargs):
    doc = get_tokenized_doc(text=text, *args, **kwargs)
    return [sent.vector for sent in doc.noun_chunks]


def text_to_sentence_vecs(text, *args, **kwargs):
    doc = get_tokenized_doc(text=text, *args, **kwargs)
    return [sent.vector for sent in doc.sents]


def text_to_vector(text, *args, **kwargs):
    doc = get_tokenized_doc(text=text, *args, **kwargs)
    return doc.vector


class OneHotGensim(BaseEstimator, TransformerMixin):
    def __init__(self, path=None):
        self.path = path
        self.id2word = None
        self.load()

    def load(self):
        if self.path is not None and os.path.exists(self.path):
            self.id2word = Dictionary.load(self.path)

    def save(self):
        if self.path is not None:
            self.id2word.save(self.path)

    def fit(self, documents, labels=None):
        self.id2word = Dictionary(documents)
        self.save()
        return self

    def transform(self, documents):
        for document in documents:
            docvec = self.id2word.doc2bow(document)
            yield sparse2full(docvec, len(self.id2word))


class WordToVec:
    def __init__(
        self,
        model: str = "en_core_web_md",
        tokenizer_kwargs: dict = None,
        vectorizer_kwargs: dict = None,
        tokenizer: Callable = Tokenizer,
    ):
        tokenizer_kwargs = {} if tokenizer_kwargs is None else tokenizer_kwargs
        vectorizer_kwargs = {} if vectorizer_kwargs is None else vectorizer_kwargs

        tokenizer_kwargs["model"] = model
        self.tokenizer = tokenizer(**tokenizer_kwargs)
        self.vectorizer = CountVectorizer(
            tokenizer=self.tokenizer.tokenize_text, **vectorizer_kwargs
        )
        self.nlp = self.tokenizer.nlp
        self._idx2vec = {}
        self._idx2word = {}
        self._word2vec = {}

    @property
    def vocabulary(self):
        try:
            return self.vectorizer.vocabulary_
        except Exception as e:
            return {}

    @property
    def idx2vec(self):
        return self._idx2vec

    @property
    def idx2word(self):
        return self._idx2word

    @property
    def word2vec(self):
        return self._word2vec

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def to_vector(self, val: [str, int]) -> np.ndarray:
        if isinstance(val, int):
            val = self.idx2word[val]
        mem = self.word2vec.get(val)
        if mem is not None:
            return mem
        return self.nlp.vocab[val]

    def add_to_vocab(self, word):
        ix = self.vocab_size + 1
        vector = self.to_vector(word)
        self.vocabulary[ix] = word
        self.idx2word[ix] = word
        self.word2vec[ix] = vector
        self.idx2vec[ix] = vector

    def clean_text(self, text, return_tokens: bool = False) -> [str, list]:
        cleaned = self.tokenizer.fit_transform(text, return_tokens=return_tokens)
        return cleaned[0] if isinstance(text, str) else cleaned

    def build_vocabulary(self, X, y=None):
        """
        Learn how to transform data based on input data, X.
        """

        self.vectorizer.fit(X)
        for word, i in self.vocabulary.items():
            vector = self.nlp.vocab[word].vector
            self._idx2word[i] = word
            self._idx2vec[i] = vector
            self._word2vec[word] = vector
        return self

    def to_onehot(self, X):
        self.vectorizer.binary = True
        return self.vectorizer.transform(X)

    def to_freq_counts(self, X):
        self.vectorizer.binary = False
        return self.vectorizer.transform(X)

    def to_word_vectors(self, X, clean_text: bool = False):
        word_seqs = []
        for text in X:
            tokens = self.clean_text(text, return_tokens=True) if clean_text else text
            vecs = [self.to_vector(token) for token in tokens]
            word_seqs.append(np.array(vecs))
        return word_seqs

    def to_seq_vectors(self, X, clean_text: bool = False):
        text = self.clean_text(X, return_tokens=False) if clean_text else X
        return [self.nlp(x).vector for x in text]

    def to_pytorch_embedding(self, pretrained: bool = True, vector_size: int = None):
        import torch
        import torch.nn as nn

        if self.idx2vec.get(0) is None and pretrained:
            raise ValueError("The vector dictionary needs to be initialize. Call fit() first.")
        elif not pretrained and vector_size is None:
            raise ValueError(
                "You need to specify an embedding size when not using pretrained weights."
            )

        vector_size = len(self.idx2vec[0]) if pretrained else vector_size
        embed = nn.Embedding(self.vocab_size, vector_size)

        if pretrained:
            # intialize the word vectors, pretrained_weights is a
            # numpy array of size (vocab_size, vector_size) and
            # pretrained_weights[i] retrieves the word vector of
            # i-th word in the vocabulary
            pretrained_weights = np.array([self.idx2vec[i] for i in range(self.vocab_size)])
            embed.weight.data.copy_(torch.from_numpy(pretrained_weights))

        return embed
