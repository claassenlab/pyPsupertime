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
    k_: int = 0
    classes_: np.array
    is_fitted_: bool = False

    @abstractmethod
    def _binary_estimator_factory():
        raise NotImplementedError()
 
    def get_binary_estimator(self):
        if self.binary_estimator_ is None:
            self.binary_estimator_  = self._binary_estimator_factory()

        if not isinstance(self.binary_estimator_, BaseEstimator):
            raise ValueError("The underlying 'binary_estimator' is not a sklearn.base.BaseEstimator. Got this instead: ", self.binary_estimator_)
        
        return self.binary_estimator_

    def _before_fit(self, data, targets):
        data, targets = check_X_y(data, targets)
        self.classes_ = np.unique(targets)
        self.k_ = len(self.classes_) - 1
        return data, targets
    
    def _after_fit(self, model):
        self.is_fitted_ = True

        # extract the thresholds and weights
        # from the 2D coefficients matrix in the sklearn model
        self.intercept_ = np.array(model.coef_[0, -self.k_:]) + model.intercept_  # thresholds
        self.coef_ = model.coef_[0, :-self.k_]   # weights

    def fit(self, data, targets, sample_weight=None):
        data, targets = self._before_fit(data, targets)

        # convert to binary problem
        data = restructure_X_to_bin(data, self.k_)
        targets = restructure_y_to_bin(targets)

        model = self.get_binary_estimator()
        
        weights = np.tile(sample_weight, self.k_) if sample_weight is not None else None
        model.fit(data, targets, sample_weight=weights)
        self._after_fit(model)

        return self

    def predict_proba(self, X):
        warnings.filterwarnings("once")

        transform = X @ self.coef_        
        logit = np.zeros(X.shape[0] * (self.k_)).reshape(X.shape[0], self.k_)
        
        # calculate logit
        for i in range(self.k_):
            # Clip exponents that are larger than MAX_EXP before np.exp for numerical stability
            # this will cause warnings and nans otherwise!
            temp = self.intercept_[i] + transform
            temp = np.clip(temp, np.min(temp), MAX_EXP)
            exp = np.exp(temp)
            logit[:, i] = exp / (1 + exp)

        prob = np.zeros(X.shape[0] * (self.k_ + 1)).reshape(X.shape[0], self.k_ + 1)
        # calculate differences
        for i in range(self.k_ + 1):
            if i == 0:
                prob[:, i] = 1 - logit[:, i]
            elif i < self.k_:
                prob[:, i] = logit[:, i-1] - logit[:, i]
            elif i == self.k_:
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

        # SGD parameters
        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.regularization = regularization
        self.loss = loss
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.power_t = power_t
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        
    def _binary_estimator_factory(self):
        return SGDClassifier(eta0 = self.eta0,
                            learning_rate = self.learning_rate,
                            max_iter = self.max_iter,
                            random_state = self.random_state,
                            alpha = self.regularization,
                            loss = self.loss,
                            penalty = self.penalty,
                            l1_ratio = self.l1_ratio,
                            fit_intercept = self.fit_intercept,
                            shuffle = self.shuffle,
                            verbose = self.verbose,
                            epsilon = self.epsilon,
                            n_jobs = self.n_jobs,
                            power_t = self.power_t,
                            validation_fraction = self.validation_fraction,
                            class_weight = self.class_weight,
                            warm_start = self.warm_start,
                            average = self.average,
                            early_stopping = self.early_stopping,
                            n_iter_no_change = self.n_iter_no_change,
                            tol = self.tol)
