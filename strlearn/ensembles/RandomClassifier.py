from ..ensembles.base import StreamingEnsemble
import numpy as np


class RandomClassifier(StreamingEnsemble):
    def __init__(self):
        pass

    def partial_fit(self, X, y, classes=None):
        return super().partial_fit(X, y, classes)

    def predict(self, X):
        return np.array([np.random.choice(self.classes_) for i in range(X.shape[0])])