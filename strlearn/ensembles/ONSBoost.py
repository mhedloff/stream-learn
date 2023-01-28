import numpy as np
from sklearn import clone

from ..ensembles import OnlineBoosting


class ONSBoost(OnlineBoosting):
    class EstimatorsParams(OnlineBoosting.EstimatorParams):
        def __init__(self):
            super().__init__()
            self._last_ok = True

        def handle_good_prediction(self, lam):
            self._last_ok = True
            return super().handle_good_prediction(lam)

        def handle_wrong_prediction(self, lam):
            self._last_ok = False
            return super().handle_wrong_prediction(lam)

        def get_prediction_factor(self):
            if self._last_ok:
                return self.calculate_prediction_element_error()
            else:
                return -self.calculate_prediction_element_error()

    def __init__(self,
                 base_estimator=None,
                 n_estimators=5,
                 update_period=100,
                 protection_period=100,
                 window_size=50):
        super().__init__(base_estimator, n_estimators)
        self.update_period = update_period
        self.protection_period = protection_period
        self.window_size = window_size
        self._window_candidate = np.empty(window_size, )
        self._window_candidate.fill(np.nan)
        self._k = 0

    def initialize_base_model_params(self, n_estimators):
        super().initialize_base_model_params(n_estimators)
        self.estimators_params = [ONSBoost.EstimatorsParams() for _ in range(self.n_estimators)]

    def process_single_sample(self, current_x, current_y):
        self._k += 1
        super().process_single_sample(current_x, current_y)
        self.handle_window_shift()

        if self._k % self.update_period == 0:
            self.handle_removal(current_x, current_y)

        return self

    def handle_window_shift(self):
        current_window_output = np.array(
            [estimator_params.get_prediction_factor() for estimator_params in self.estimators_params]
        )
        self._window_candidate[:self.window_size - 1] = self._window_candidate[1:self.window_size]
        self._window_candidate[self.window_size - 1] = np.argmin(current_window_output) \
            if np.sum(current_window_output) < 0 else np.nan

    def handle_removal(self, current_x, current_y):
        defined_candidates = self._window_candidate[~np.isnan(self._window_candidate)]
        if len(defined_candidates) > 0:
            values, counts = np.unique(defined_candidates, return_counts=True)
            candidate_i = np.int(values[np.argmax(counts)])
            if self.estimators_params[candidate_i].get_age() >= self.protection_period:
                # make place for new base model
                self.ensemble_.pop(candidate_i)
                self.estimators_params.pop(candidate_i)

                # create & fit new base model
                new_base_model = clone(self.base_estimator)
                new_model_params = ONSBoost.EstimatorsParams()

                r = max(1, np.random.poisson(1))
                new_base_model.partial_fit(current_x, current_y, self.classes_, sample_weight=r)
                new_model_params.handle_good_prediction(1)

                # append new base model to the end of the list
                self.ensemble_.append(new_base_model)
                self.estimators_params.append(new_model_params)
