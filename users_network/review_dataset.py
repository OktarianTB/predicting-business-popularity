import pandas as pd

class ReviewDataset:
    def __init__(self, review_file_path, k=5):
        self.reviews_df = pd.read_json(review_file_path, lines=True)
        self.reviews_df = self.reviews_df.set_index("business_id")
        self.business_ids = self.reviews_df.index.unique()

        split_ids = self._train_val_test_split()

        self.data = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.popularity = {
            "train": [],
            "val": [],
            "test": [],
        }
        for mode in ["train", "val", "test"]:
            for bid in split_ids[mode]:
                _df = self.reviews_df.loc[bid].sort_values(by="date")[:k]
                _data = {
                    "business_id": bid,
                    "t1": _df["date"][0],
                    "t2": _df["date"][-1],
                    "texts": _df["text"].tolist(),  # TODO: need to tokenize first.
                    "user_ids": _df["user_id"].tolist(),
                }
                self.data[mode].append(_data)
                popularity_metric = 0.  # get real popularity_metric;
                self.popularity[mode].append(popularity_metric)

    def _train_val_test_split(self):
        raise NotImplementedError

    def get_batch(self, batch_size=16, mode="train"):
        for i in range(0, len(self.data[mode]), batch_size):
            yield self.data[mode][i:min(i+batch_size, len(self.data[mode]))], \
                self.popularity[mode][i:min(i+batch_size, len(self.data[mode]))]
        