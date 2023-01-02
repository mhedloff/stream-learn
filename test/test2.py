import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

from strlearn.ensembles import ONSBoost, UOB, RandomClassifier, OnlineBoosting
from strlearn.evaluators import TestThenTrain
from strlearn.streams import StreamGenerator
from strlearn.metrics import balanced_accuracy_score

import warnings

warnings.filterwarnings("ignore")

N_CHUNKS = 100
N_FEATURES = 5

ons_boost = ONSBoost(n_estimators=5, base_estimator=GaussianNB(), p=100, k=200)
uob = UOB(base_estimator=GaussianNB(), n_estimators=5)
random_cls = RandomClassifier()
onb = OnlineBoosting(n_estimators=5, base_estimator=GaussianNB())

ensembles = [
    {'name': 'onsboost', 'clf': ons_boost},
    {'name': 'uob', 'clf': uob},
    {'name': 'random', 'clf': random_cls},
    {'name': 'onb', 'clf': onb}
]

clfs = (ons_boost, uob, random_cls, onb)
metrics = (balanced_accuracy_score,)

stream = StreamGenerator(n_chunks=N_CHUNKS, n_features=N_FEATURES, weights=(2, 5, 0.9))
ttt = TestThenTrain(metrics=metrics, verbose=True)

ttt.process(stream, clfs)
ons_scores = ttt.scores[0, :, 0]
uob_scores = ttt.scores[1, :, 0]
random_scores = ttt.scores[2, :, 0]
onb_scores = ttt.scores[3, :, 0]

x = np.linspace(0, N_CHUNKS - 1, N_CHUNKS - 1)
plt.figure(figsize=(18, 9))

for j, metric in enumerate(metrics):
    print(metric.__name__)
    for i, clff in enumerate(ensembles):
        plt.plot(x, ttt.scores[i, :, j], label=clff['name'])

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pass
