from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import os.path
from tqdm import tqdm
import logging
from scipy import spatial


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class ContentBasedRecommender:
    def __init__(self, reader, vector_size, min_word_count, training_epochs):
        self.corpus_id = reader.corpus_name
        self.reader = reader
        self.min_word_count = min_word_count
        self.vector_size = vector_size
        self.training_epochs = training_epochs
        self.model_specs_str = f"doc2vec_c{self.corpus_id}_v{vector_size}_w{min_word_count}_e{training_epochs}"
        self.model = None
        self.corpus = None
        self.id2index = None
        self.similarity_cache = {}
        self.corpus_size = 0

    def train_or_load(self):
        default_file = f"models/{self.model_specs_str}.model"
        if os.path.isfile(default_file):
            self.load(default_file)
        else:
            self.train()

    def train(self, save=True):
        self.corpus, self.id2index = self.reader.read()
        self.corpus_size = len(self.corpus)
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_word_count,
            epochs=self.training_epochs,
        )
        self.model.build_vocab(self.corpus)
        self.model.train(
            self.corpus,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )
        if save:
            self.model.save(f"models/{self.model_specs_str}.model")

    def load(self, file):
        self.model = Doc2Vec.load(file)

    def predict(self, predict_item, items_evaluated, items_ratings):
        sims_sum = 0
        sims_prod = 0
        for item_id, item_rating in zip(items_evaluated, items_ratings):
            sim = self.__get_similarity(predict_item, item_id)
            sims_prod += sim * item_rating
            sims_sum += sim
        return sims_prod / max(sims_sum, 0.0000001)

    def __get_similarity(self, id1, id2):
        key = f"{id1}\f{id2}"
        if key in self.similarity_cache:
            return self.similarity_cache[key]

        doc1 = self.model.docvecs[id1]
        doc2 = self.model.docvecs[id2]
        sim = spatial.distance.cosine(doc1, doc2)
        self.similarity_cache[key] = sim
        inv_key = f"{id2}\f{id1}"
        self.similarity_cache[inv_key] = sim
        return sim

    def score(self, users_data, predictions_reader, item_averages):
        err = 0
        err2 = 0
        counter = 0
        for user_id, item_id, target_rating in predictions_reader.read_lines(
            desc="Calculating score"
        ):
            if user_id in users_data:
                user_items = users_data[user_id].items
                user_ratings = users_data[user_id].ratings
                pred = (
                    self.predict(item_id, user_items, user_ratings)
                    + users_data[user_id].mean
                )
                err2 += (pred - target_rating) ** 2
                err += abs(pred - target_rating)
            else:
                err += (item_averages[item_id] - target_rating) ** 2
            counter += 1
        return {"rmse": (err2 / counter) ** (1 / 2), "mae": (err / counter)}

    def most_similar(self, document_data):
        if self.corpus is None:
            self.corpus, self.id2index = self.reader.read()
            self.corpus_size = len(self.corpus)
        x = self.model.infer_vector(document_data)
        sim = self.model.docvecs.most_similar([x], topn=1)
        return self.corpus[self.id2index[sim[0][0]]]
