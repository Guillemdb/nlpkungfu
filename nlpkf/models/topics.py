from typing import Callable
import pandas as pd
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from umap import UMAP
import pyLDAvis
import pyLDAvis.sklearn
from nlpkf.preprocessing.corpus import CorpusProcessor
from nlpkf.preprocessing.tokenizer import Tokenizer


class TopicAnalizer(CorpusProcessor):
    def __init__(
        self,
        n_components: int,
        model: Callable = LatentDirichletAllocation,
        tokenizer: Callable = Tokenizer,
        model_params=None,
        *args,
        **kwargs
    ):
        """
        Args:
            n_components: Number of topics that will be modelled.
            model: Model used to perform the topic modelling. Must implement
                fit_transform()
            tokenizer: Callable that returns a Tokenizer Object
            model_params: parameters for the model, passed as kwargs.
            *args: args of the parent class CorpusProcessor.
            **kwargs: kwargs of the parent class CorpusProcessor.
        """
        model_params = model_params if model_params is not None else {}
        super(TopicAnalizer, self).__init__(tokenizer=tokenizer, *args, **kwargs)
        self.model = model(n_components=n_components, **model_params)

    def corpus_to_dataset(self, corpus, *args, **kwargs):
        return self.vectorizer.transform(corpus)

    def fit(self, corpus, y=None):
        self.build_vocabulary(corpus, y=y)
        dataset = self.corpus_to_dataset(corpus=corpus)
        preds = self.model.fit_transform(dataset)
        return preds

    def print_topics(self, top_n=10):
        for idx, topic in enumerate(self.model.components_):
            print("Topic %d:" % idx)
            print(
                [
                    (self.vectorizer.get_feature_names()[i], "{:.2f}".format(topic[i]))
                    for i in topic.argsort()[: -top_n - 1 : -1]
                ]
            )

    def plot_words(self, vectorized_corpus, height=600, width=600, *args, **kwargs):
        svd = UMAP(n_components=2, *args, **kwargs)
        words_2d = svd.fit_transform(vectorized_corpus.T)

        df = pd.DataFrame(columns=["x", "y", "word"])
        df["x"], df["y"], df["word"] = (
            words_2d[:, 0],
            words_2d[:, 1],
            self.vectorizer.get_feature_names(),
        )

        source = ColumnDataSource(ColumnDataSource.from_df(df))
        labels = LabelSet(
            x="x",
            y="y",
            text="word",
            y_offset=8,
            text_font_size="8pt",
            text_color="#555555",
            source=source,
            text_align="center",
        )

        plot = figure(
            plot_width=height,
            plot_height=width,
            title="Word embeddings",
            x_axis_label="SVD component 1",
            y_axis_label="SVD component 2",
        )
        plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
        plot.add_layout(labels)
        show(plot, notebook_handle=True)

    @staticmethod
    def plot_documents(vectorized_corpus, width=600, height=600, *args, **kwargs):
        svd = UMAP(n_components=2, *args, **kwargs)
        documents_2d = svd.fit_transform(vectorized_corpus)

        df = pd.DataFrame(columns=["x", "y", "document"])
        df["x"], df["y"], df["document"] = (
            documents_2d[:, 0],
            documents_2d[:, 1],
            range(vectorized_corpus.shape[0]),
        )

        source = ColumnDataSource(ColumnDataSource.from_df(df))
        labels = LabelSet(
            x="x",
            y="y",
            text="document",
            y_offset=8,
            text_font_size="8pt",
            text_color="#555555",
            source=source,
            text_align="center",
        )

        plot = figure(
            plot_width=width,
            plot_height=height,
            title="Document embeddings",
            x_axis_label="SVD component 1",
            y_axis_label="SVD component 2",
            title_location="above",
        )
        plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
        plot.add_layout(labels)
        show(plot, notebook_handle=True)

    def visualize_topics(self, vectorized_corpus):
        return pyLDAvis.sklearn.prepare(self.model, vectorized_corpus, self.vectorizer)
