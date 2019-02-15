import nltk
import torch
import torch.nn as nn
import numpy as np
from nlpkf.preprocessing.vectorizers import WordToVec
from nlpkf.utils import device


class CorpusProcessor(WordToVec):
    def __init__(self, *args, **kwargs):
        super(CorpusProcessor, self).__init__(*args, **kwargs)
        self.text_corpus = []
        self.vector_dim = 300  # TODO: find a way to calculate this on the fly

    def clean_corpus(self, corpus, *args, **kwargs):
        return [self.clean_text(doc, *args, **kwargs) for doc in corpus]

    def tokenize_corpus(self, corpus=None):
        data = corpus if corpus is not None else self.text_corpus
        return (self.clean_text(doc, True) for doc in data)
        return (doc.split(" ") for doc in data)

    def tokens_to_index(self, corpus):
        index_values = [[self.vocabulary[token] for token in sentence] for sentence in corpus]
        return index_values

    def to_ngrams(self, corpus, *args, **kwargs):
        if isinstance(corpus, str):
            tokens = corpus.split(" ")
        else:
            tokens = list(self.tokenize_corpus(corpus)) if isinstance(corpus[0], str) else corpus
        return [list(nltk.ngrams(doc, *args, **kwargs)) for doc in tokens]

    def to_pytorch_embedding(
        self, pretrained: bool = True, vector_size: int = None, device=device
    ):
        if self.idx2vec.get(0) is None and pretrained:
            raise ValueError("The vector dictionary needs to be initialize. Call fit() first.")
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
