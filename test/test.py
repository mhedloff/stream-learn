import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from strlearn.ensembles import UOB, ONSBoost
from strlearn.evaluators import TestThenTrain
from strlearn.metrics import balanced_accuracy_score
from strlearn.streams import StreamGenerator

stream = StreamGenerator(n_drifts=3)
ons_boost = ONSBoost(base_estimator=GaussianNB(), n_estimators=3)
uob = UOB(base_estimator=GaussianNB(), n_estimators=3)
clfs = (ons_boost, uob)
evaluator = TestThenTrain(metrics=(balanced_accuracy_score))

evaluator.process(stream, clfs)
print(evaluator.scores)

x = np.linspace(0, 249, 249)
plt.plot(x, evaluator.scores[0, :, 1])
plt.plot(x, evaluator.scores[1, :, 1])
plt.show()
