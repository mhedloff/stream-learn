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

    def __init__(self, base_estimator=None, n_estimators=5, k=100, p=100, w=50):
        super().__init__(base_estimator, n_estimators)
        self.K = k
        self.p = p
        self.w = w
        self.window_candidate = np.empty(w, )
        self.window_candidate.fill(np.nan)
        self.k = 0

    def initialize_base_model_params(self, n_estimators):
        super().initialize_base_model_params(n_estimators)
        self.estimators_params = [ONSBoost.EstimatorsParams() for _ in range(self.n_estimators)]

    def process_single_sample(self, current_x, current_y):
        self.k += 1
        super().process_single_sample(current_x, current_y)
        self.handle_window_shift()

        if self.k % self.K == 0:
            self.handle_removal(current_x, current_y)

        return self

    def handle_window_shift(self):
        current_window_output = np.array(
            [estimator_params.get_prediction_factor() for estimator_params in self.estimators_params]
        )
        self.window_candidate[:self.w - 1] = self.window_candidate[1:self.w]
        self.window_candidate[self.w - 1] = np.argmin(current_window_output) \
            if np.sum(current_window_output) < 0 else np.nan

    def handle_removal(self, current_x, current_y):
        not_protected_models = [estimator_params for estimator_params in self.estimators_params
                                if estimator_params.get_age() >= self.p]
        defined_candidates = self.window_candidate[~np.isnan(self.window_candidate)]
        if len(defined_candidates) > 0:
            values, counts = np.unique(defined_candidates, return_counts=True)
            candidate_i = values[np.argmax(counts)]
            if candidate_i in not_protected_models:
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
