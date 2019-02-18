import string
import nltk
import spacy
from typing import Callable
from nltk.corpus import stopwords
from nlpkf.utils import STOPWORDS_NLTK, STOPWORDS_SPACY, EOS_TOKEN, SOS_TOKEN, normalize_string


def clean_text(
    text, return_tokens=False, return_tokenizer: bool = False, *args, **kwargs
) -> [str, tuple]:
    tokenizer = Tokenizer(*args, **kwargs)
    data = tokenizer.fit_transform(text, return_tokens=return_tokens)[0]
    return data, tokenizer if return_tokenizer else data


class BaseTokenizer:
    def __init__(
        self,
        word_tokenize: Callable,
        stem_func: Callable = None,
        stop_words: [list, tuple, set] = None,
        lemma_func: Callable = None,
        use_lemma: bool = False,
        use_stems: bool = False,
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        remove_nums: bool = False,
        to_lowercase: bool = False,
        language="english",
        normalize_strings: bool = False,
        *args,
        **kwargs,
    ):
        self.normalize_strings = normalize_strings
        self.use_stems = use_stems
        self.remove_stopwords = remove_stopwords
        self.remove_nums = remove_nums
        self.language = language
        self.remove_punctuation = remove_punctuation
        self.to_lowercase = to_lowercase
        self.stem = stem_func
        self.use_lemma = use_lemma
        self.lemma = lemma_func
        self.word_tokenize = word_tokenize
        self.stop_words = stop_words if stop_words is not None else []
        self.special_tokens = set((str(SOS_TOKEN), str(EOS_TOKEN)))

    def is_punctuation(self, word):
        return word in string.punctuation

    def is_stop_word(self, word):
        return word in self.stop_words

    def is_number(self, word):
        return False

    def is_special_token(self, word):
        return str(word).upper() in self.special_tokens

    def tokenize_text(self, text):

        text = text.lower() if self.to_lowercase else text
        for i, token in enumerate(self.word_tokenize(text)):
            # Special tokens are not processed
            if self.is_special_token(token):
                yield str(token).upper()
                continue
            # Filter tokens
            is_punct = self.remove_punctuation and self.is_punctuation(token)
            is_blank = len(str(token).strip()) == 0
            is_stop = self.remove_stopwords and self.is_stop_word(token)
            is_num = self.remove_nums and self.is_number(token)
            if is_blank or is_punct or is_stop or is_punct or is_num:
                continue
            # Perform lemmatization and stemming
            if self.use_lemma and self.lemma is not None:
                val = self.lemma(token)
            elif self.use_stems:
                val = self.stem(token)
            else:
                val = token
            # Discard empty strings after normalizing
            val = str(val) if not self.normalize_strings else normalize_string(str(val))
            val = val.strip()
            if len(val) == 0:
                continue
            yield val

    def fit_transform(self, data, return_tokens: bool = True):
        corpus = self.data_as_corpus(data)
        if return_tokens:
            return [[token for token in self.tokenize_text(text)] for text in corpus]
        else:
            return [" ".join(self.tokenize_text(text)) for text in corpus]

    @staticmethod
    def data_as_corpus(data: [str, list, tuple, set]):
        if isinstance(data, list) and isinstance(data[0], str):
            return data
        elif isinstance(data, str):
            return [data]
        try:
            if isinstance(data[0], str):
                return list(data)
            else:
                raise ValueError("Data {} does not contain strings".format(data))
        except IndexError:
            raise ValueError("Data {} does not contain strings".format(data))


class TokenizerNltk(BaseTokenizer):
    def __init__(
        self,
        use_stems: bool = False,
        remove_stopwords: bool = False,
        language="english",
        remove_punctuation: bool = True,
        to_lowercase: bool = True,
        use_lemma: bool = False,
        *args,
        **kwargs,
    ):
        stem_func = nltk.stem.SnowballStemmer(language).stem
        word_tokenize = nltk.word_tokenize
        stop_words = set(stopwords.words(language)) if remove_stopwords else set()
        lemma_func = nltk.stem.WordNetLemmatizer().lemmatize

        super(TokenizerNltk, self).__init__(
            stem_func=stem_func,
            use_stems=use_stems,
            lemma_func=lemma_func,
            stop_words=stop_words,
            word_tokenize=word_tokenize,
            remove_stopwords=remove_stopwords,
            remove_punctuation=remove_punctuation,
            to_lowercase=to_lowercase,
            use_lemma=use_lemma,
            language=language,
        )


class TokenizerSpacy(BaseTokenizer):
    def __init__(
        self,
        use_stems: bool = False,
        remove_stopwords: bool = False,
        language="english",
        remove_punctuation: bool = True,
        to_lowercase: bool = False,
        use_lemma: bool = False,
        model: str = "en_core_web_md",
        remove_nums: bool = False,
        nlp=None,
        *args,
        **kwargs,
    ):
        if nlp is None:
            nlp = spacy.load(model)
        word_tokenize = lambda x: nlp(x)

        stop_words = nlp.Defaults.stop_words if remove_stopwords else set()
        lemma_func = lambda x: x.lemma_

        super(TokenizerSpacy, self).__init__(
            use_stems=use_stems,
            stop_words=stop_words,
            word_tokenize=word_tokenize,
            lemma_func=lemma_func,
            remove_stopwords=remove_stopwords,
            remove_punctuation=remove_punctuation,
            to_lowercase=to_lowercase,
            language=language,
            use_lemma=use_lemma,
            remove_nums=remove_nums,
        )
        self.nlp = nlp

    def is_punctuation(self, token):
        return token.is_punct

    def is_stop_word(self, token):
        return token.norm_ in self.stop_words

    def is_number(self, word):
        return word.is_digit

    def is_special_token(self, word):
        val = str(word).upper() in self.special_tokens
        return val


class DualTokenizer(BaseTokenizer):
    def __init__(
        self,
        mode: str = "Spacy",
        tokenize_mode: str = None,
        stop_words_mode: str = None,
        lemma_mode: str = None,
        stop_words: [list, set] = None,
        *args,
        **kwargs,
    ):
        self.mode = mode.lower()
        self.word_tokenize_mode = tokenize_mode if tokenize_mode is not None else self.mode
        self.stop_words_mode = stop_words_mode if stop_words_mode is not None else self.mode
        self.lemma_func_mode = lemma_mode if lemma_mode is not None else self.mode

        self.spacy = TokenizerSpacy(*args, **kwargs)
        self.nlp = self.spacy.nlp
        self.nltk = TokenizerNltk(*args, **kwargs)

        word_tok_mode = mode.lower() if tokenize_mode is None else tokenize_mode.lower()
        word_tokenize = (
            self.spacy.word_tokenize
            if word_tok_mode.lower() == "spacy"
            else self.nltk.word_tokenize
        )
        # External stopwords override predefined
        if stop_words is None:
            stop_words_mode = mode.lower() if stop_words_mode is None else stop_words_mode.lower()
            stop_words = (
                self.spacy.stop_words
                if stop_words_mode.lower() == "spacy"
                else self.nltk.stop_words
            )

        lemma_mode = mode.lower() if lemma_mode is None else lemma_mode.lower()
        lemma_func = self.spacy.lemma if lemma_mode.lower() == "spacy" else self.nltk.lemma

        proc_kwargs = {
            "stem_func": self.nltk.stem,
            "word_tokenize": word_tokenize,
            "stop_words": stop_words,
            "lemma_func": lemma_func,
        }
        proc_kwargs.update(kwargs)
        super(DualTokenizer, self).__init__(*args, **proc_kwargs)

    def is_punctuation(self, token):
        return (
            self.spacy.is_punctuation(token)
            if self.word_tokenize_mode == "spacy"
            else self.nltk.is_punctuation(token)
        )

    def is_stop_word(self, token):
        return (
            self.spacy.is_stop_word(token)
            if self.word_tokenize_mode == "spacy"
            else self.nltk.is_stop_word(token)
        )

    def is_number(self, token):
        return (
            self.spacy.is_number(token)
            if self.word_tokenize_mode == "spacy"
            else self.nltk.is_number(token)
        )

    def is_special_token(self, token):
        return (
            self.spacy.is_special_token(token)
            if self.word_tokenize_mode == "spacy"
            else self.nltk.is_special_token(token)
        )


class Tokenizer(DualTokenizer):
    def __init__(
        self,
        filter_pos: [list, bool] = True,
        stop_words=None,
        language="english",
        use_stems: bool = False,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        to_lowercase: bool = True,
        use_lemma: bool = True,
        remove_nums: bool = True,
        *args,
        **kwargs,
    ):
        stop_words = stop_words if stop_words is not None else []
        stop_words = list(STOPWORDS_NLTK) + list(STOPWORDS_SPACY) + stop_words
        stop_words = set(stop_words)
        super(Tokenizer, self).__init__(
            use_stems=use_stems,
            stop_words=stop_words,
            remove_stopwords=remove_stopwords,
            remove_punctuation=remove_punctuation,
            to_lowercase=to_lowercase,
            language=language,
            use_lemma=use_lemma,
            remove_nums=remove_nums,
            mode="spacy",
            *args,
            **kwargs,
        )
        if filter_pos:
            self.filter_pos = True
            self.valid_pos = (
                ["NOUN", "VERB", "ADJ"] if isinstance(filter_pos, bool) else filter_pos
            )
        else:
            self.filter_pos = False
            self.valid_pos = []

    def tokenize_text(self, text):

        text = text.lower() if self.to_lowercase else text
        for i, token in enumerate(self.word_tokenize(text)):
            # Special tokens are not processed
            if self.is_special_token(token):
                yield str(token).upper()
                continue
            # Filter tokens
            is_punct = self.remove_punctuation and self.is_punctuation(token)
            is_blank = len(str(token).strip()) == 0
            is_stop = self.remove_stopwords and self.is_stop_word(token)
            is_num = self.remove_nums and self.is_number(token)
            not_pos = self.filter_pos and self.discard_by_pos(token)
            if is_blank or is_punct or is_stop or is_punct or is_num or not_pos:
                continue
            # Perform lemmatization and stemming
            if self.use_lemma and self.lemma is not None:
                val = self.lemma(token)
            elif self.use_stems:
                val = self.stem(token)
            else:
                val = token
            # Discard empty strings after normalizing
            val = str(val) if not self.normalize_strings else normalize_string(str(val))
            val = val.strip()
            if len(val) == 0:
                continue
            yield val

    def discard_by_pos(self, token):
        return token.pos_ not in self.valid_pos

    def is_stop_word(self, token):
        return (
            token.norm_ in self.stop_words
            if self.word_tokenize_mode == "spacy"
            else token in self.stop_words
        )
