from typing import List, Generator
import nltk
import torch
import torch.nn as nn
import numpy as np
from nlpkf.preprocessing.vectorizers import WordToVec
from nlpkf.utils import device


class CorpusProcessor(WordToVec):
    """The CorpusProcessor extends the functionality of the WordToVec class by
     allowing it to take a corpus as input, instead of single strings.
     """

    def __init__(self, *args, **kwargs):
        super(CorpusProcessor, self).__init__(*args, **kwargs)
        self.vector_dim = 300  # TODO: find a way to calculate this on the fly

    def clean_corpus(self, corpus: List[str], *args, **kwargs) -> [List[List[str]], List[str]]:
        """
        Takes a list of raw documents and processes them according to the rules
        of the current Tokenizer being used.
        Args:
            corpus: List of documents to be processed.
            *args: args passed to clean_text.
            **kwargs: kwargs passed to clean_text.

        Returns:
            The processed corpus as a list of strings or a corpus of tokens.
        """
        return [self.clean_text(doc, *args, **kwargs) for doc in corpus]

    def tokenize_corpus(self, corpus: List[str]) -> Generator:
        """
        Returns a generator that returns a list of preprocessed tokens for
        each document
        Args:
            corpus: List of documents represented as strings.

        Returns:
            Generator of lists of tokens.
        """
        return (self.clean_text(doc, True) for doc in corpus)
        return (doc.split(" ") for doc in corpus)

    def tokens_to_index(self, corpus: List[List[str]]) -> List[List[int]]:
        """
        Transform a corpus represented by lists of tokens to a corpus represented
        by a list of integers. These integers represent the vocabulary key of
        each token in the input corpus.

        Args:
            corpus: Corpus as a list of tokens for every document.

        Returns:
            The corpus as a list of indexes for every document.

        """
        index_values = [[self.vocabulary[token] for token in sentence] for sentence in corpus]
        return index_values

    def to_ngrams(self, corpus: [List, str], *args, **kwargs) -> list:
        """
        Transforms a string or sequence of items (tokens, indexes or vectors)
        into a list of N-grams.
        Args:
            corpus: Input corpus to be transformed.
            *args: args passed to the nltk.ngrams() function
            **kwargs: kwargs passed to the nltk.ngrams() function

        Returns:
            A list of processed documents, where each processed documents is a
             list of tuples. Each tuple represents one ngram that can be extracted
             from that document.
        """
        if isinstance(corpus, str):
            tokens = corpus.split(" ")
        else:
            tokens = list(self.tokenize_corpus(corpus)) if isinstance(corpus[0], str) else corpus
        return [list(nltk.ngrams(doc, *args, **kwargs)) for doc in tokens]

    def to_pytorch_embedding(
        self, pretrained: bool = True, vector_size: int = None, device=device
    ) -> torch.nn.Embedding:
        """
        Create a Pytorch Embedding from the current vocabulary of the CorpusProcessor.
        this embedding can use pretrained weights imported from the vector representation
        that Spacy offers.
        Args:
            pretrained: If true, the weights are initialized to the vector embedding
                provided by Spacy.
            vector_size: In case no pretraining is used, it represents the dimension
                of the embedding that will be created.
            device: Device where the Embedding model will be run.

        Returns:
            tuple containing the embedding module created and its embedding size.
        """
        if self.idx2vec.get(0) is None and pretrained:
            raise ValueError("The vector dictionary needs to be initialized. Call fit() first.")
        elif not pretrained and vector_size is None:
            raise ValueError(
                "You need to specify an embedding size when not using pretrained weights."
            )

        vector_size = len(self.idx2vec[0]) if pretrained else vector_size
        embed = nn.Embedding(self.vocab_size, vector_size).to(device)

        if pretrained:
            # intialize the word vectors, pretrained_weights is a
            # numpy array of size (vocab_size, vector_size) and
            # pretrained_weights[i] retrieves the word vector of
            # i-th word in the vocabulary
            pretrained_weights = np.array([self.idx2vec[i] for i in range(self.vocab_size)])
            embed.weight.data.copy_(torch.from_numpy(pretrained_weights).to(device))

        return embed, vector_size
