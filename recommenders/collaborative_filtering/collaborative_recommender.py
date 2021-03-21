from LibRecommender.libreco.evaluation import evaluate

try:
    from LibRecommender.libreco.data import DatasetPure, random_split
    from LibRecommender.libreco.algorithms import SVDpp
except Exception:
    pass


class CollaborativeRecommender:
    def __init__(self, data, multi_ratio=None):
        if multi_ratio is None:
            multi_ratio = [0.64, 0.16, 0.2]
        train_data, eval_data, test_data = random_split(data, multi_ratios=multi_ratio)
        self._train_data, data_info = DatasetPure.build_trainset(train_data)
        self._eval_data = DatasetPure.build_evalset(eval_data)
        self._test_data = DatasetPure.build_testset(test_data)
        print(data_info)

        self._model = SVDpp(
            task="rating",
            data_info=data_info,
            embed_size=16,
            n_epochs=5,
            lr=0.001,
            reg=None,
            batch_size=256
        )

    def fit(self):
        self._model.fit(self._train_data, verbose=2, eval_data=self._eval_data, metrics=["rmse", "mae", "r2"])

    def evaluate(self):
        print("evaluate_result: ", evaluate(model=self._model, data=self._test_data, metrics=["rmse", "mae"]))