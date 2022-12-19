import numpy as np
from sklearn import clone

from ..ensembles.base import StreamingEnsemble


class OnlineBoosting(StreamingEnsemble):
    class EstimatorParams:
        def __init__(self):
            self._age = 0
            self._l_sc = 0
            self._l_sw = 0
            self._error = 0

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

        def calculate_prediction_element_error(self):
            return np.log((1 - self._error) / self._error)

        def get_age(self):
            return self._age

        def get_error(self):
            return self._error

    def __init__(self, base_estimator=None, n_estimators=5):
        super().__init__(base_estimator, n_estimators)
        self.estimators_params = [OnlineBoosting.EstimatorParams() for _ in range(n_estimators)]

    def predict(self, X):
        yy = np.zeros(shape=(X.shape[0], len(self.classes_)))
        predictions_per_clf = np.array([base_model.predict(X) for base_model in self.ensemble_]).T
        for p, current_sample_predictions in enumerate(predictions_per_clf):
            for m, base_model_prediction in enumerate(current_sample_predictions):
                yy[p, base_model_prediction] += self.estimators_params[m].calculate_prediction_element_error()
        return np.argmax(yy, axis=-1).reshape(-1)

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self
        if len(self.ensemble_) == 0:
            self.ensemble_ = [
                clone(self.base_estimator) for i in range(self.n_estimators)
            ]
        for i in range(len(self.X_)):
            self.process_single_sample(self.X_[i].reshape(1, -1), [self.y_[i]])

        return self

    def process_single_sample(self, current_x, current_y):
        lambda_value = 1
        for w, base_model in enumerate(self.ensemble_):
            r = max(1, np.random.poisson(lambda_value))
            current_model = self.estimators_params[w]
            base_model.partial_fit(current_x, current_y, self.classes_, sample_weight=r)

            predicted_class = base_model.predict(current_x)
            lambda_value = current_model.handle_good_prediction(lambda_value) \
                if predicted_class == current_y else current_model.handle_wrong_prediction(lambda_value)
            current_model.increment_age()
        return lambda_value
