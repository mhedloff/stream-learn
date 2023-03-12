import numpy as np
from sklearn import clone

from ..ensembles.base import StreamingEnsemble


class OnlineBoosting(StreamingEnsemble):
    class EstimatorParams:
        def __init__(self):
            self._age = 0
            self._l_sc = 0
            self._l_sw = 0
            self._error = 1

        def handle_good_prediction(self, lam):
            self._l_sc += lam
            self._error = (self._l_sw + 1) / (self._l_sc + self._l_sw + 1)
            return .5 * lam / (1 - self._error)

        def handle_wrong_prediction(self, lam):
            self._l_sw += lam
            self._error = (self._l_sw + 1) / (self._l_sc + self._l_sw + 1)
            return .5 * lam / self._error

        def increment_age(self):
            self._age += 1

        def increment_age_by(self, r):
            self._age += r

        def calculate_prediction_element_error(self):
            return np.log((1 - self._error) / self._error)

        def get_age(self):
            return self._age

        def get_error(self):
            return self._error

    def __init__(self, base_estimator=None, n_estimators=5):
        super().__init__(base_estimator, n_estimators)
        self.estimators_params = None
        self.initialize_base_model_params(n_estimators)

    def initialize_base_model_params(self, n_estimators):
        self.estimators_params = [OnlineBoosting.EstimatorParams() for _ in range(n_estimators)]

    def predict(self, X):
        y = np.zeros(shape=(X.shape[0], len(self.classes_)))
        if X.shape[0] == 0:
            print(X.shape)
            print(X)
            print('kurwa ja już nie dam rady z tym jebanym garbage collectorem')
            pass
        predictions_per_clf = np.array([base_model.predict(X) for base_model in self.ensemble_]).T
        for p, current_sample_predictions in enumerate(predictions_per_clf):
            for m, base_model_prediction in enumerate(current_sample_predictions):
                y[p, base_model_prediction] += self.estimators_params[m].calculate_prediction_element_error()
        return np.argmax(y, axis=-1).reshape(-1)

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self
        if len(self.ensemble_) == 0:
            self.ensemble_ = [clone(self.base_estimator) for _ in range(self.n_estimators)]
        for i in range(len(self.X_)):
            self.process_single_sample(self.X_[i].reshape(1, -1), [self.y_[i]])

        return self

    def process_single_sample(self, current_x, current_y):
        lambda_value = 1
        for w, base_model in enumerate(self.ensemble_):
            current_model = self.estimators_params[w]
            r = np.random.poisson(lambda_value)
            if r > 0:
                for i in range(r):
                    base_model.partial_fit(current_x, current_y, self.classes_)
                current_model.increment_age_by(1)

                predicted_class = base_model.predict(current_x)
                if predicted_class == current_y:
                    lambda_value = current_model.handle_good_prediction(lambda_value)
                else:
                    lambda_value = current_model.handle_wrong_prediction(lambda_value)
        return lambda_value
