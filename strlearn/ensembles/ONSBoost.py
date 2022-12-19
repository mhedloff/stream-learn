import numpy as np
from sklearn import clone

from ..ensembles import OnlineBoosting


class ONSBoost(OnlineBoosting):
    def __init__(self, base_estimator=None, n_estimators=5, k=100, p=50, with_replacement=True):
        super().__init__(base_estimator, n_estimators)
        self.with_replacement = with_replacement
        self.K = k
        self.p = p
        self.rejected_base_models = []
        self.rejected_base_models_params = []
        self.k = 0

    def process_single_sample(self, current_x, current_y):
        self.k += 1
        super().process_single_sample(current_x, current_y)
        if self.k % self.K == 0:
            not_protected_models = [estimator for estimator in self.estimators_params
                                    if estimator.get_age() >= self.p]
            if len(not_protected_models) > 0:
                candidate = np.argmax([not_protected_estimator.get_error()
                                       for not_protected_estimator in not_protected_models])

                rejected_model = self.ensemble_.pop(candidate)
                rejected_model_params = self.estimators_params.pop(candidate)

                if self.with_replacement and len(self.rejected_base_models) > 0:
                    new_base_model = self.rejected_base_models.pop()
                    new_base_model_params = self.rejected_base_models_params.pop()
                    self.ensemble_.append(new_base_model)
                    self.estimators_params.append(new_base_model_params)
                else:
                    new_base_model = clone(self.base_estimator)
                    self.ensemble_.append(new_base_model)
                    new_model_params = OnlineBoosting.EstimatorParams()
                    self.estimators_params.append(new_model_params)
                    new_base_model.partial_fit(current_x, current_y, self.classes_)
                    new_model_params.handle_good_prediction(1)
                self.rejected_base_models.append(rejected_model)
                self.rejected_base_models_params.append(rejected_model_params)

        return self
