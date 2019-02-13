from typing import Callable
from nlpkf.data_processing import DataProcessor
from nlpkf.tokenizer import TopicTokenizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD


class TopicAnalizer(DataProcessor):
    def __init__(
        self,
        n_components,
        model: Callable=LatentDirichletAllocation,
        tokenizer: Callable=TopicTokenizer,
        model_params=None,
        *args,
        **kwargs
    ):
        model_params = model_params if model_params is not None else {}
        super(TopicAnalizer, self).__init__(tokenizer=tokenizer, *args, **kwargs)
        self.model = model(n_components=n_components, **model_params)

    def fit(self, X, y=None):
        clean_corpus = self.build_vocabulary(X, y=y)
        dataset = self.vectorizer.fit_transform(clean_corpus)
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
