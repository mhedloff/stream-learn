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

ons_boost = ONSBoost(n_estimators=5, with_replacement=False, base_estimator=GaussianNB(), p=50, k=200)
uob = UOB(base_estimator=GaussianNB(), n_estimators=5)
random_cls = RandomClassifier()
# random_cls = ONSBoost(n_estimators=5, with_replacement=True, base_estimator=GaussianNB(), p=50, k=200)
onb = OnlineBoosting(n_estimators=5, base_estimator=GaussianNB())

clfs = (ons_boost, uob, random_cls, onb)
stream = StreamGenerator(n_chunks=N_CHUNKS, weights=(2, 5, 0.9))
ttt = TestThenTrain(metrics=(balanced_accuracy_score,), verbose=True)

ttt.process(stream, clfs)
ons_scores = ttt.scores[0, :, 0]
uob_scores = ttt.scores[1, :, 0]
random_scores = ttt.scores[2, :, 0]
onb_scores = ttt.scores[3, :, 0]

x = np.linspace(0, N_CHUNKS - 1, N_CHUNKS - 1)
plt.figure(figsize=(18, 9))
plt.plot(x, uob_scores, label='uob')
plt.plot(x, ons_scores, label='ons')
plt.plot(x, random_scores, label='rand')
plt.plot(x, onb_scores, label='onb')
plt.legend()
plt.tight_layout()
plt.show()

if __name__ == '__main__':
    pass
