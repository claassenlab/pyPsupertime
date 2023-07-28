from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import anndata as ad
import numpy as np
import pandas as pd
from numpy.random import default_rng
import warnings

from .preprocessing import restructure_X_to_bin, restructure_y_to_bin


# Maximum positive number before numpy 64bit float overflows in np.exp()
MAX_EXP = 709


class PsupertimeBaseModel(ClassifierMixin, BaseEstimator, ABC):
    binary_estimator_: BaseEstimator = None
    regularization: float
    random_state: int = 1234
    coef_: np.array
    intercept_: np.array
    k: int = 0
    classes_: np.array
    is_fitted_: bool = False

    def _get_binary_estimator(self):
        if not isinstance(self.binary_estimator_, BaseEstimator):
            raise ValueError("'binary_estimator_ not initialized or is not a 'sklearn.base.BaseEstimator'")
        
        return self.binary_estimator_

    def _before_fit(self, data, targets):
        data, targets = check_X_y(data, targets)
        self.classes_ = np.unique(targets)
        self.k = len(self.classes_) - 1
        return data, targets
    
    def _after_fit(self, model):
        self.is_fitted_ = True

        # extract the thresholds and weights
        # from the 2D coefficients matrix in the sklearn model
        self.intercept_ = np.array(model.coef_[0, -self.k:]) + model.intercept_  # thresholds
        self.coef_ = model.coef_[0, :-self.k]   # weights

    def fit(self, data, targets, sample_weight=None):
        data, targets = self._before_fit(data, targets)

        # convert to binary problem
        data = restructure_X_to_bin(data, self.k)
        targets = restructure_y_to_bin(targets)

        model = self._get_binary_estimator()
        
        weights = np.tile(sample_weight, self.k) if sample_weight is not None else None
        model.fit(data, targets, sample_weight=weights)
        self._after_fit(model)

        return self

    def predict_proba(self, X):
        warnings.filterwarnings("once")

        transform = X @ self.coef_        
        logit = np.zeros(X.shape[0] * (self.k)).reshape(X.shape[0], self.k)
        
        # calculate logit
        for i in range(self.k):
            # Clip exponents that are larger than MAX_EXP before np.exp for numerical stability
            # this will cause warnings and nans otherwise!
            temp = self.intercept_[i] + transform
            temp = np.clip(temp, np.min(temp), MAX_EXP)
            exp = np.exp(temp)
            logit[:, i] = exp / (1 + exp)

        prob = np.zeros(X.shape[0] * (self.k + 1)).reshape(X.shape[0], self.k + 1)
        # calculate differences
        for i in range(self.k + 1):
            if i == 0:
                prob[:, i] = 1 - logit[:, i]
            elif i < self.k:
                prob[:, i] = logit[:, i-1] - logit[:, i]
            elif i == self.k:
                prob[:, i] = logit[:, i-1]
        
        warnings.filterwarnings("always")
        return prob
    
    def predict(self, X):
        return np.apply_along_axis(np.argmax, 1, self.predict_proba(X))

    def predict_psuper(self, anndata: ad.AnnData, inplace=True):
        
        transform = anndata.X @ self.coef_
        predicted_labels = self.predict(anndata.X)      

        if inplace:
            anndata.obs["psupertime"] = transform
            anndata.obs["predicted_label"] = predicted_labels
        
        else:
            return pd.DataFrame({"psupertime": transform,
                                 "predicted_label": predicted_labels},
                                 index=anndata.obs.index.copy())
    
    def gene_weights(self, anndata: ad.AnnData, inplace=True):
        if inplace:
            anndata.var["psupertime_weight"] = self.coef_
        else:
            return pd.DataFrame({"psupertime_weight": self.coef_},
                                index=anndata.var.index.copy())
        
    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return metrics.mean_absolute_error(pred, y, sample_weight=sample_weight)


class BaselineSGDModel(PsupertimeBaseModel):
    
    def __init__(self, 
                 max_iter=100, 
                 random_state=1234, 
                 regularization=0.01, 
                 n_iter_no_change=5, 
                 early_stopping=True,
                 tol=1e-3,
                 learning_rate="optimal",
                 eta0=0,
                 loss='log_loss', 
                 penalty='elasticnet', 
                 l1_ratio=1, 
                 fit_intercept=True, 
                 shuffle=True, 
                 verbose=0, 
                 epsilon=0.1, 
                 n_jobs=1, 
                 power_t=0.5, 
                 validation_fraction=0.1,
                 class_weight=None,
                 warm_start=False,
                 average=False):
        
        # underlying binary logistic model
        self.binary_estimator_ = SGDClassifier(eta0 = eta0,
                                               learning_rate = learning_rate,
                                               max_iter = max_iter,
                                               random_state = random_state,
                                               alpha = regularization,
                                               loss = loss,
                                               penalty = penalty,
                                               l1_ratio = l1_ratio,
                                               fit_intercept = fit_intercept,
                                               shuffle = shuffle,
                                               verbose = verbose,
                                               epsilon = epsilon,
                                               n_jobs = n_jobs,
                                               power_t = power_t,
                                               validation_fraction = validation_fraction,
                                               class_weight = class_weight,
                                               warm_start = warm_start,
                                               average = average,
                                               early_stopping = early_stopping,
                                               n_iter_no_change = n_iter_no_change,
                                               tol = tol)

        # fitting/data parameters
        self.k = None
        self.intercept_ = []
        self.coef_ = []

        # random
        self.rng = default_rng(seed=self.random_state)
