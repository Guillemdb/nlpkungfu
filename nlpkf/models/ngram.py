from typing import Callable
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nlpkf.preprocessing.corpus import CorpusProcessor
from nlpkf.utils import device


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int, hidden_size: int):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.to(device)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class NgramModel:
    """The NgramModel learns the next element of an n-gram sequence. It uses a
    NGramLanguageModeler to predict the next word of a sequence, and it
    incorporates all the logic needed to process raw text.

    """

    def __init__(
        self,
        context_size: int,
        dataproc: Callable = CorpusProcessor,
        embedding_dim=None,
        load_embedding: bool = True,
        optimizer=None,
        hidden_size: int = 128,
        *args,
        **kwargs
    ):
        """
        Args:
            context_size:
            dataproc:
            embedding_dim:
            load_embedding:
            optimizer:
            hidden_size:
        """
        self.dataproc = dataproc(*args, **kwargs)
        self.vocab_size = self.dataproc.vocab_size
        self.embedding_dim = embedding_dim if embedding_dim is not None else dataproc.vector_dim
        self.load_embedding = load_embedding
        self.context_size = context_size
        self.model = None
        self.optimizer = optimizer
        self._array_to_words = np.vectorize(lambda x: self.dataproc.idx2word[x])
        self.hidden_size = hidden_size

    def array_to_words(self, data):
        return self._array_to_words(data)

    def build_vocabulary(self, corpus, *args, **kwargs):
        self.dataproc.build_vocabulary(corpus, *args, **kwargs)
        self.vocab_size = self.dataproc.vocab_size

    def init_model(self):

        self.embedding_dim = len(self.dataproc.to_vector(0))
        self.model = NGramLanguageModeler(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            context_size=self.context_size,
            hidden_size=self.hidden_size,
        )
        optimizer = lambda x: optim.SGD(x, lr=0.001) if self.optimizer is None else self.optimizer
        self.optimizer = optimizer(self.model.parameters())
        if self.load_embedding:
            self.model.embeddings, _ = self.dataproc.to_pytorch_embedding(pretrained=True)

    def corpus_to_dataset(self, corpus) -> tuple:
        tokens = self.dataproc.tokenize_corpus(corpus=corpus)
        indexes = self.dataproc.tokens_to_index(tokens)
        ngram_ix = self.dataproc.to_ngrams(indexes, self.context_size + 1)
        dataset = np.array(
            [list(seq) for sentence in ngram_ix for seq in sentence], dtype=np.int64
        )
        X, y = dataset[:, :-1], dataset[:, -1]
        return X, y

    def train(self, X, y, n_epochs: int = 1):
        losses = []
        loss_function = nn.NLLLoss()
        for epoch in tqdm(range(n_epochs)):
            total_loss = 0
            for context, target in zip(X, y):
                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                context_idxs = torch.tensor([context], dtype=torch.long, device=device)

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                self.model.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = self.model(context_idxs)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                loss = loss_function(
                    log_probs, torch.tensor([target], dtype=torch.long, device=device)
                )

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                self.optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
            losses.append(total_loss)
        return losses

    def fit(self, corpus, n_epochs: int = 10):
        print("Building vocabulary.")
        self.build_vocabulary(corpus=corpus)
        self.init_model()
        print("Preprocessing dataset.")
        X, y = self.corpus_to_dataset(corpus)
        print("Training.")
        losses = self.train(X=X, y=y, n_epochs=n_epochs)
        return X, y, losses

    def predict(self, text, return_dataset: bool = False):
        tokens = self.dataproc.clean_text(text, return_tokens=True)
        indexes = self.dataproc.tokens_to_index([tokens])
        ngram_ix = self.dataproc.to_ngrams(indexes, self.context_size)
        dataset = np.array(
            [list(seq) for sentence in ngram_ix for seq in sentence], dtype=np.int64
        )
        preds = [
            self.model(torch.tensor([x], dtype=torch.long, device=device)).argmax(1).item()
            for x in dataset
        ]

        return (preds, dataset) if return_dataset else preds

    def evaluate_pred(self, text: str):
        preds, data = self.predict(text, return_dataset=True)
        preds = self.array_to_words(preds)
        data = self.array_to_words(data)
        text = "{}\n".format(text)
        text += "   In   |   Pred   \n"
        for d, p in zip(data, preds):
            text += " {} | {}\n".format(" ".join(d), p)
        print(text)
