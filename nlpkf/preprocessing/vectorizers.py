import os
import numpy as np
from typing import Callable, List
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from nlpkf.preprocessing.tokenizer import Tokenizer, clean_text


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
    """The WordToVec includes everything you need to transform raw text into
    vector representations suitable for using with different machine learning
    models.

    It uses a Tokenizer to preprocess the raw text, and an sklearn CountVectorizer
    to build the vocabulary and process the text input.

    This class keeps track of the vocabulary, the mapping between words, indexes,
    and the vector representations offered by spacy.
    """

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
        self.tokenizer = (
            tokenizer(**tokenizer_kwargs) if not isinstance(tokenizer, Tokenizer) else tokenizer
        )
        lower_vect = vectorizer_kwargs.get("lowercase", False)
        vectorizer_kwargs["lowercase"] = lower_vect
        self.vectorizer = CountVectorizer(
            tokenizer=self.tokenizer.tokenize_text, **vectorizer_kwargs
        )
        self.nlp = self.tokenizer.nlp
        self._idx2vec = {}
        self._idx2word = {}
        self._word2vec = {}

    @property
    def vocabulary(self):
        """Access the vocabulary dictionary containing the mapping of the
        vocabulary words to its respective indexes.
        """
        try:
            return self.vectorizer.vocabulary_
        except Exception as e:
            return {}

    @property
    def idx2vec(self) -> dict:
        """Access the  dictionary containing the mapping of the
        word indexes to its respective vector representations.
        """
        return self._idx2vec

    @property
    def idx2word(self) -> dict:
        """Access the  dictionary containing the mapping of the
        word indexes to its respective word representations.
        """
        return self._idx2word

    @property
    def word2vec(self) -> dict:
        """Access the  dictionary containing the mapping of the
        word indexes to its respective vector representations.
        """
        return self._word2vec

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocabulary)

    def to_vector(self, val: [str, int]) -> np.ndarray:
        """Map and integer representing a word index or a string representing a
        word to its respective vector representation.
        """
        if isinstance(val, int):
            val = self.idx2word[val]
        mem = self.word2vec.get(val)
        if mem is not None:
            return mem
        return self.nlp.vocab[val].vector

    def add_to_vocab(self, word):
        """Add a word to the current vocabulary."""
        ix = self.vocab_size + 1
        vector = self.to_vector(word)
        self.vocabulary[ix] = word
        self.idx2word[ix] = word
        self.word2vec[ix] = vector
        self.idx2vec[ix] = vector

    def clean_text(self, text: str, return_tokens: bool = False) -> [str, List[str]]:
        """
        Use the Tokenizer to clean a string.
        Args:
            text: String to be cleaned
            return_tokens: Whether to return the result as a list of tokens or as
                an string.

        Returns:
            Cleaned text either as a string or as a list of tokens.

        """
        cleaned = self.tokenizer.fit_transform(text, return_tokens=return_tokens)
        return cleaned[0] if isinstance(text, str) else cleaned

    def build_vocabulary(self, X: List[str], y=None):
        """
        Fit the Vectorizer to build the vocabulary on a given corpus, and update
        the mapping dictionaries accordingly.
        Args:
            X: Corpus of raw text as a list of strings.
            y: Not used. Implemented to match sklearn interface.

        Returns:
            None
        """

        self.vectorizer.fit(X)
        for word, i in self.vocabulary.items():
            vector = self.nlp.vocab[word].vector
            self._idx2word[i] = word
            self._idx2vec[i] = vector
            self._word2vec[word] = vector

    def to_onehot(self, X: List[str]) -> np.ndarray:
        """
        Transform a given corpus to its one hot encoded vector representation.
        Args:
            X: Corpus to be transformed

        Returns:
            One hot representation of the corpus.

        """
        self.vectorizer.binary = True
        return self.vectorizer.transform(X)

    def to_freq_counts(self, X: List[str]) -> np.ndarray:
        """
        Transform a given corpus to its word frequency counts representation.
        Args:
            X: Corpus to be transformed

        Returns:
            Bag of words representation of the corpus.

        """
        self.vectorizer.binary = False
        return self.vectorizer.transform(X)

    def to_word_vectors(self, X: List[str], clean_text: bool = False) -> list:
        """
        Transform a given corpus to its word vector representation.
        Args:
            X: Corpus to be transformed

        Returns:
            Word vector representation of the corpus.

        """
        word_seqs = []
        for text in X:
            tokens = self.clean_text(text, return_tokens=True) if clean_text else text
            vecs = [self.to_vector(token) for token in tokens]
            word_seqs.append(np.array(vecs))
        return word_seqs

    def to_seq_vectors(self, X: List[str], clean_text: bool = False) -> list:
        """
        Transform a corpus to a document vector representation, where each document
        is represented by a dense vector.
        Args:
            X: List of documents to be transformed
            clean_text:

        Returns:

        """
        text = self.clean_text(X, return_tokens=False) if clean_text else X
        return [self.nlp(x).vector for x in text]
