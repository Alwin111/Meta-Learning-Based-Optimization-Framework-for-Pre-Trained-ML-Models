import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class MetaLearningModel:

    def __init__(self):
        self.model = RandomForestClassifier()
        self.encoder = LabelEncoder()
        self.trained = False

    def train(self):
        df = pd.read_csv("experiments/meta_dataset.csv")

        df["model_type"] = self.encoder.fit_transform(df["model_type"])
        self.known_classes = list(self.encoder.classes_)

        X = df[["model_type", "latency", "model_size"]]
        y = df["optimization"]

        self.model.fit(X, y)
        self.trained = True

    def recommend(self, model_type, latency, model_size):

        if not self.trained:
            raise Exception("Meta model not trained")

        if model_type not in self.known_classes:
            model_type = self.known_classes[0]

        model_type_encoded = self.encoder.transform([model_type])[0]

        features = [[model_type_encoded, latency, model_size]]

        return self.model.predict(features)[0]
