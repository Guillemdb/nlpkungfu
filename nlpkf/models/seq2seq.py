import random
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Callable
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from nlpkf.preprocessing.corpus import CorpusProcessor
from nlpkf.utils import EOS_TOKEN, SOS_TOKEN, device, time_since


MAX_LENGTH = 100


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, embedding_size, hidden_size: int):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.compressed = nn.Linear(embedding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.to(device)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        compressed = F.softmax(self.compressed(embedded), dim=1)
        output = compressed
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, embedding_size: int):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.to(device)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        embedding_size: int,
        dropout_p=0.1,
        max_length=MAX_LENGTH,
    ):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)

        self.attn = nn.Linear(self.embedding_size + self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.to(device)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.embedding_size, device=device)


class Seq2Seq:
    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        encoder_optimizer=None,
        decoder_optimizer=None,
        criterion=None,
        max_length=MAX_LENGTH,
        teacher_forcing_ratio: float = 0.5,
    ):
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion
        self.max_length = max_length
        self.plot_losses = []

    @staticmethod
    def sentences_to_index(corpus: List[str], processor: CorpusProcessor):

        tokens_corpus = processor.tokenize_corpus(corpus)
        index_corpus = processor.tokens_to_index(tokens_corpus)
        return index_corpus

    def init_model(
        self, input_size, output_size, hidden_size, max_length, embedding_size, dropout=0.1
    ):
        self.encoder = EncoderRNN(
            input_size=input_size, hidden_size=hidden_size, embedding_size=embedding_size
        ).to(device)
        self.decoder = AttnDecoderRNN(
            output_size=output_size,
            hidden_size=hidden_size,
            max_length=max_length,
            embedding_size=embedding_size,
        ).to(device)
        self.criterion = nn.NLLLoss()

    def init_optimizers(self, optimizer: Callable = optim.SGD, lr: float = 0.01, *args, **kwargs):
        self.encoder_optimizer = optimizer(self.encoder.parameters(), lr=lr, *args, **kwargs)
        self.decoder_optimizer = optimizer(self.decoder.parameters(), lr=lr, *args, **kwargs)

    def encode_inputs(self, input_tensor, encoder_hidden=None):
        encoder_hidden = self.encoder.init_hidden() if encoder_hidden is None else encoder_hidden
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)
        input_length = input_tensor.size(0)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden.to(device)
            )
            encoder_outputs[ei] = encoder_output[0, 0]
        return encoder_outputs, encoder_hidden

    def train_on_pair(self, input_tensor, target_tensor, SOS_TOKEN_ix, EOS_TOKEN_ix):

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        target_length = target_tensor.size(0)
        loss = 0
        decoder_input = torch.tensor([[SOS_TOKEN_ix]], device=device)

        encoder_outputs, decoder_hidden = self.encode_inputs(input_tensor)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_TOKEN_ix:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def train_on_corpus(
        self,
        input_vectors,
        target_vectors,
        SOS_TOKEN_ix: int,
        EOS_TOKEN_ix: int,
        n_iters: int = 1,
        print_every=1000,
        plot_every=100,
    ):
        self.plot_losses = []
        start = time.time()
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        # training_pairs = [tensorsFromPair(random.choice(pairs))
        #                  for i in range(n_iters)]
        for epoch_num in range(1, n_iters + 1):

            for i in tqdm(range(len(target_vectors))):

                input_tensor = torch.tensor(
                    input_vectors[i], dtype=torch.long, device=device
                ).view(-1, 1)
                target_tensor = torch.tensor(
                    target_vectors[i], dtype=torch.long, device=device
                ).view(-1, 1)

                loss = self.train_on_pair(
                    input_tensor,
                    target_tensor,
                    SOS_TOKEN_ix=SOS_TOKEN_ix,
                    EOS_TOKEN_ix=EOS_TOKEN_ix,
                )
                print_loss_total += loss
                plot_loss_total += loss
                if i % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    self.plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

            if epoch_num % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                pct_done = epoch_num / n_iters * 100
                print(
                    "%s epoch: %d  complete: %d%% loss: %.4f"
                    % (time_since(start, epoch_num / n_iters), epoch_num, pct_done, print_loss_avg)
                )

        return self.plot_losses, plot_loss_total

    def predict(
        self,
        sentence,
        SOS_TOKEN_ix,
        EOS_TOKEN_ix,
        max_length,
        src_proc: CorpusProcessor,
        target_proc: CorpusProcessor,
    ):
        with torch.no_grad():
            input_tensor = self.sentences_to_index([sentence], processor=src_proc)
            input_tensor = torch.tensor(input_tensor, dtype=torch.long, device=device).view(-1, 1)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.init_hidden()

            encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_TOKEN_ix]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_TOKEN_ix:
                    decoded_words.append(EOS_TOKEN)
                    break
                else:
                    decoded_words.append(target_proc.idx2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[: di + 1]

    def evaluate(self, sentence, target=None, *args, **kwargs):
        print(">", sentence)
        if target is not None:
            print("=", target)
        output_words, attentions = self.predict(sentence, *args, **kwargs)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")

    @staticmethod
    def plot_attention(input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap="bone")
        fig.colorbar(cax)
        xticks = [""] + input_sentence.split(" ")
        yticks = [""] + output_words
        # Set up axes
        ax.set_xticklabels(xticks, rotation=90)
        ax.set_yticklabels(yticks)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        return ax

    def evaluate_attention(self, input_sentence, *args, **kwargs):
        output_words, attentions = self.predict(input_sentence, *args, **kwargs)
        print("input =", input_sentence)
        print("output =", " ".join(output_words))
        self.plot_attention(input_sentence, output_words, attentions)
        return input_sentence, output_words

    def plot_loss(self):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(self.plot_losses)
        plt.show()
        return ax


def add_tokens_to_sentence(text, sos_token: str = SOS_TOKEN, eos_token: str = EOS_TOKEN):
    return "{} {} {}".format(sos_token, text, eos_token)


def preprocess_translator(text, proc_func, sos_token: str = SOS_TOKEN, eos_token: str = EOS_TOKEN):
    source = []
    target = []
    for src, dst in proc_func(text):

        src, dst = (
            add_tokens_to_sentence(src),
            add_tokens_to_sentence(dst, sos_token=sos_token, eos_token=eos_token),
        )
        source.append(src)
        target.append(dst)
    return source, target


def preprocess_catalan(
    file_location: str = "cat.txt", sos_token: str = SOS_TOKEN, eos_token: str = EOS_TOKEN
):
    with open(file_location, "r") as file:
        data = file.read()
    src, dst = preprocess_translator(
        data,
        lambda x: [tuple(x.split("\t")) for x in str(data).split("\n")[:-1]],
        sos_token=sos_token,
        eos_token=eos_token,
    )
    return src, dst


class Translator(Seq2Seq):
    def __init__(
        self,
        hidden_size: int,
        embedding_size: int = None,
        processor: Callable = CorpusProcessor,
        pretrained: bool = False,
        learning_rate: float = 0.001,
        teacher_forcing_ratio: float = 0.5,
        *args,
        **kwargs
    ):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.pretrained = pretrained

        self.src_proc = processor(*args, **kwargs)
        self.target_proc = processor(tokenizer=self.src_proc.tokenizer, *args, **kwargs)

        super(Translator, self).__init__(teacher_forcing_ratio=teacher_forcing_ratio)

    @staticmethod
    def get_max_seq_len(*vectors_list):
        return max([max([len(x) for x in corpus]) for corpus in vectors_list])

    def init_model(self, input_size, output_size, max_length, *args, **kwargs):

        encoder_embedding, _ = self.src_proc.to_pytorch_embedding(
            pretrained=self.pretrained, vector_size=self.embedding_size
        )
        decoder_embedding, self.embedding_size = self.target_proc.to_pytorch_embedding(
            pretrained=self.pretrained, vector_size=self.embedding_size
        )
        super(Translator, self).init_model(
            input_size=input_size,
            output_size=output_size,
            hidden_size=self.hidden_size,
            max_length=max_length,
            embedding_size=self.embedding_size,
        )

        self.encoder.embedding = encoder_embedding
        self.decoder.embedding = decoder_embedding

    def build_vocabularies(self, src_corpus, target_corpus, *args, **kwargs):
        self.src_proc.build_vocabulary(src_corpus, *args, **kwargs)
        self.target_proc.build_vocabulary(target_corpus, *args, **kwargs)
        input_size = self.src_proc.vocab_size
        output_size = self.target_proc.vocab_size
        return input_size, output_size

    def train(
        self,
        x_corpus,
        y_corpus,
        n_iters: int = 1,
        print_every: int = 1000,
        plot_every: int = 1000,
        preprocess: bool = False,
    ):
        if preprocess:
            x_corpus = self.sentences_to_index(x_corpus, processor=self.src_proc)
            y_corpus = self.sentences_to_index(y_corpus, processor=self.target_proc)
        return self.train_on_corpus(
            x_corpus,
            y_corpus,
            n_iters=n_iters,
            print_every=print_every,
            plot_every=plot_every,
            SOS_TOKEN_ix=self.target_proc.vocabulary[SOS_TOKEN],
            EOS_TOKEN_ix=self.target_proc.vocabulary[EOS_TOKEN],
        )

    def fit(
        self,
        src_corpus,
        target_corpus,
        n_iters: int = 1,
        print_every: int = 1000,
        plot_every: int = 1000,
        train: bool = False,
        *args,
        **kwargs
    ):
        print("Building vocabulary.")
        input_size, output_size = self.build_vocabularies(
            src_corpus, target_corpus, *args, **kwargs
        )
        print("Converting to tensors.")
        x_corpus = self.sentences_to_index(src_corpus, processor=self.src_proc)
        y_corpus = self.sentences_to_index(target_corpus, processor=self.target_proc)

        self.max_length = self.get_max_seq_len(x_corpus, y_corpus)
        print("Initializing model and optimizers.")
        self.init_model(input_size=input_size, output_size=output_size, max_length=self.max_length)
        self.init_optimizers(optimizer=optim.RMSprop, lr=self.learning_rate)
        if train:
            print("training model")
            self.train(
                x_corpus, y_corpus, n_iters=n_iters, print_every=print_every, plot_every=plot_every
            )
        return x_corpus, y_corpus

    def predict(self, sentence: str, *args, **kwargs):

        return super(Translator, self).predict(
            sentence,
            SOS_TOKEN_ix=self.target_proc.vocabulary[SOS_TOKEN],
            EOS_TOKEN_ix=self.target_proc.vocabulary[EOS_TOKEN],
            max_length=self.max_length,
            src_proc=self.src_proc,
            target_proc=self.target_proc,
        )

    def evaluate(self, sentence, target=None, *args, **kwargs):
        print(">", sentence)
        if target is not None:
            print("=", target)
        output_words, attentions = self.predict(sentence)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")
