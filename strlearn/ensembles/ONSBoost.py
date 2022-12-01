import numpy as np
from numpy import argmax
from sklearn import clone

from ..ensembles.base import StreamingEnsemble


class ONSBoost(StreamingEnsemble):
    def __init__(self, base_estimator=None, n_estimators=5, look_back_range=100, protection_period=50):
        super().__init__(base_estimator, n_estimators)
        self.ensemble_ = []
        self.look_back_range = look_back_range
        self.protection_period = protection_period
        self.estimators_params = [{"age": 0, "lambda_correct": 0, "lambda_wrong": 0, "error": 1., "initialised": False}
                                  for i in range(n_estimators)]
        self.k = 0


    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self
        if len(self.ensemble_) == 0:
            self.ensemble_ = [
                clone(self.base_estimator) for i in range(self.n_estimators)
            ]
        lambda_value = 1
        self.k += 1
        for i in range(len(self.X_)):
            for w, base_model in enumerate(self.ensemble_):
                r = np.random.poisson(lambda_value)
                current_estimator = self.estimators_params[w]
                current_x = self.X_[i].reshape(1, -1)
                current_y = [self.y_[i]]
                for _ in range(r):
                    base_model.partial_fit(current_x, current_y, self.classes_)
                    current_estimator['initialised'] = True
                predicted_class = 0
                if current_estimator['initialised']:
                    try:
                        predicted_class = base_model.predict(current_x)
                    except:
                        print(current_x, '->', current_y, '::', w, self.estimators_params[w])
                if predicted_class == current_y:
                    current_estimator['lambda_correct'] += lambda_value
                    current_estimator['error'] = (current_estimator['lambda_wrong'] + 1) / \
                                                 (1 + current_estimator['lambda_correct'] + current_estimator['lambda_wrong'])
                    lambda_value /= (2 * (1 - current_estimator['error']))
                else:
                    current_estimator['lambda_wrong'] += lambda_value
                    current_estimator['error'] = (current_estimator['lambda_wrong'] + 1) / \
                                                 (1 + current_estimator['lambda_correct'] + current_estimator['lambda_wrong'])
                    lambda_value /= (2 * (current_estimator['error']))
                current_estimator['age'] += 1

        if self.k >= self.protection_period:
            self.k = 0
            not_protected_estimators = [estimator for estimator in self.estimators_params
                                        if estimator['age'] >= self.protection_period]
            if len(not_protected_estimators) == 0:
                return self
            candidate = np.argmax([not_protected_estimator['error'] for not_protected_estimator in not_protected_estimators])
            self.ensemble_.pop(candidate)
            self.ensemble_.insert(candidate, clone(self.base_estimator))
            self.estimators_params[candidate] = {"age": 0, "lambda_correct": 0, "lambda_wrong": 0, "error": 1., "initialised": False}


        return self

