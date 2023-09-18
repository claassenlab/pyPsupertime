from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from scipy import sparse
import torch
import anndata as ad
import numpy as np
import pandas as pd
from numpy.random import default_rng
import warnings

from .preprocessing import restructure_X_to_bin, restructure_y_to_bin, transform_labels


# Maximum positive number before numpy 64bit float overflows in np.exp()
MAX_EXP = 709


class LogisticRegression(torch.nn.Module):

    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class PsupertimeBaseModel(ClassifierMixin, BaseEstimator, ABC):
    """
    Abstract Base class to build scikit-learn compatible models for PyPsupertime derived from `sklearn.base.BaseEstimator` and 
    `sklearn.base.ClassifierMixin`.

    Provides methods for restructuring ordinal data into a binary representation and 
    for fitting a nested binary logistic classifier.
    
    Provides predict methods, that uses the fitted binary classifier to estimate the probabilities and labels of
    the ordinal multiclass problem.

    :ivar method: Statistical model used for ordinal logistic regression: One of `"proportional"`, `"forward"` 
     and `"backward"`, corresponding to cumulative proportional odds, forward continuation ratio and
     backward continuation ratio.
    :type method: str 
    :ivar regularization: parmeter controlling the sparsity of the model. Wrapper for the respective parameter
     of the nested `binary_estimator_`. Not necessary
    :type regularization: float
    :ivar k_: number of thresholds to be learned, equal to one less than the number of unique ordinal labels 
    :type k_: int

    """
    method: str = "proportional"
    regularization: float
    random_state: int = 1234
    coef_: np.array
    intercept_: np.array
    k_: int = 0
    classes_: np.array
    is_fitted_: bool = False

    def _before_fit(self, data, targets, sample_weights=None):
        data, targets = check_X_y(data, transform_labels(targets), accept_sparse=True)
        self.classes_ = np.unique(targets)
        self.k_ = len(self.classes_) - 1

        try:
            if sample_weights is not None:
                if not len(sample_weights) == len(targets):
                    raise ValueError("The parameter sample_weight has incompatible weight with the target vector. Shape: %s Expected: %s" % (len(sample_weights), len(targets)))

        except TypeError as e:
            print(e)
            raise ValueError("The parameter sample_weights has no length. Received: %s" % sample_weights)
        
        return data, targets
    
    def _after_fit(self, model):
        self.is_fitted_ = True

        # extract the thresholds and weights
        # from the 2D coefficients matrix in the sklearn model
        self.intercept_ = np.array(model.coef_[0, -self.k_:]) + model.intercept_  # thresholds
        self.coef_ = model.coef_[0, :-self.k_]   # weights

    @abstractmethod
    def fit(self, data, targets, sample_weight=None):
        """Template fit function for derived models.

        :param data: 2d data
        :type data: numpy or numpy.sparse matrix
        :param targets: Array-like object with ordinal labels
        :type targets: Iterable
        :param sample_weight: label weights to be used for training and scoring, defaults to None
        :type sample_weight: Iterable, optional
        :return: fitted estimator
        :rtype: PsupertimeBaseModel
        """
        pass

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
            obs_copy = anndata.obs.copy()
            obs_copy["psupertime"] = transform
            obs_copy["predicted_label"] = predicted_labels
            return obs_copy
    
    def gene_weights(self, anndata: ad.AnnData, inplace=True):
        if inplace:
            anndata.var["psupertime_weight"] = self.coef_
        else:
            return pd.DataFrame({"psupertime_weight": self.coef_},
                                index=anndata.var.index.copy())


class BatchSGDModel(PsupertimeBaseModel):
    """
    BatchSGDModel is a classifier derived from `PsupertimBaseModel` that wraps an `SGDClassifier`
    as logistic binary estimator.
    
    It overwrites the superclass `_binary_estimator_factory() and `fit()` methods. The latter is wrapping
    the `SGDClassifier.partial_fit()` function to fit the model in batches for a reduced memory footprint.
    
    """
    def __init__(self,
                 method="proportional",
                 early_stopping_batches=False,
                 n_batches=1,
                 max_iter=1000, 
                 random_state=1234, 
                 regularization=0.01, 
                 n_iter_no_change=5, 
                 early_stopping=True,
                 tol=1e-3,
                 learning_rate=0.1,
                 penalty='elasticnet', 
                 l1_ratio=0.75, 
                 shuffle=True, 
                 verbosity=0, 
                 epsilon=0.1, 
                 validation_fraction=0.1,
                 class_weight=None):

        self.method = method

        # model hyperparameters
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.regularization = regularization
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.shuffle = shuffle
        self.verbosity = verbosity
        self.epsilon = epsilon
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.n_batches = n_batches
        self.early_stopping_batches = early_stopping_batches

        # data attributes
        self.k_ = None
        self.intercept_ = []
        self.coef_ = []

    def _binary_estimator_factory(self):
        return None

    def fit(self, X, y, sample_weights=None):
        """Fit ordinal logistic model. 
        Multiclass data is converted to binarized representation and one weight per feature, 
        as well as a threshold for each class is fitted with a binary logistic classifier.

        Derived from a `sklearn.linear.SGDClassifier`, fitted in batches according to `self.n_batches` 
        for reduced memory usage.
        

        :param X: Data as 2d-matrix
        :type X: numpy.array or scipy.sparse
        :param y: ordinal labels
        :type y: Iterable
        :param sample_weight: Label weights for fitting and scoring, defaults to None. Can be used for example for class balancing.
        :type sample_weight: Iterable, optional
        :return: fitted classifier
        :rtype: BatchSGDModel
        """
        rng = np.random.default_rng(self.random_state)
        X, y = self._before_fit(X, y, sample_weights)

        if self.early_stopping:
            # TODO: This is a full copy of the input data -> split an index array instead and work with slices?
            X, X_test, y, y_test = train_test_split(X, y, test_size=self.validation_fraction, stratify=y, random_state=rng.integers(9999))
            
            # TODO: initializing binarized matrices for testing can be significant memory sink!
            y_test_bin = restructure_y_to_bin(y_test)
            del(y_test)

            if self.early_stopping_batches:
                n_test = X_test.shape[0]
                test_indices = np.arange(len(y_test_bin))
            else:
                X_test_bin = restructure_X_to_bin(X_test, self.k_)
                del(X_test)
        
        # diagonal matrix, to construct the binarized X per batch
        thresholds = np.identity(self.k_)
        if sparse.issparse(X):
            thresholds = sparse.csr_matrix(thresholds)

        n = X.shape[0]
        n_features = X.shape[1] + self.k_

        # Logistic regression model, defined as a perceptron
        model = LogisticRegression(input_dim=n_features)

        # Adaptive momentum SGD optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Loss function: Binary Cross Entropy = Log loss
        criterion = torch.nn.BCELoss()

        # Sample weight to balance classes of y
        if sample_weights is None:
            sample_weights = torch.Tensor(np.ones_like(y))
        else:
            sample_weights = torch.Tensor(sample_weights)

        # Mask for applying penalty: Only apply to gene features, don't apply to thresholds
        penalty_mask = torch.Tensor(np.concatenate((np.ones(X.shape[1]), np.zeros(self.k_))))

        # binarize only the labels already
        y_bin = torch.Tensor(restructure_y_to_bin(y))
        
        # create an index array and shuffle
        sampled_indices = rng.integers(len(y_bin), size=len(y_bin))

        # iterations over all data
        epoch = 0

        # tracking previous scores for early stopping
        best_score = - np.inf
        n_no_improvement = 0

        while epoch < self.max_iter:

            epoch += 1

            start = 0
            for i in range(1, self.n_batches+1):
                end = (i * len(y_bin) // self.n_batches)
                batch_idx = sampled_indices[start:end]
                batch_idx_mod_n = batch_idx % n
                
                if sparse.issparse(X):
                    # TODO: Fix sparsity! Converting to dense format is a hack to get this to work
                    X_batch = torch.Tensor(sparse.hstack((X[batch_idx_mod_n], thresholds[batch_idx // n])).todense())
                else:
                    X_batch = torch.Tensor(np.hstack((X[batch_idx_mod_n,:], thresholds[batch_idx // n])))
                
                y_batch = y_bin[batch_idx]
                start = end
                sample_weights_batch = sample_weights[batch_idx_mod_n]

                # Set stored gradients to zero
                optimizer.zero_grad()

                # Forward pass
                outputs = model(X_batch)

                # calculate parameter penalties
                model_weights = list(model.parameters())[0]
                l1_term = torch.norm(penalty_mask * model_weights, 1)  # **2 TODO: Find out why/if squared ... 
                #l2_term = torch.norm(model_params, 2) ** 2

                # calculate loss with sample_weights and penalty
                loss = criterion(torch.squeeze(outputs), y_batch)
                loss = loss + self.regularization * l1_term

                # backward pass to calculate gradients
                loss.backward()
                
                # update weights
                optimizer.step()

            # Early stopping using the test data 
            if False: #self.early_stopping:  # TODO: Disabled for now

                # build test data in batches as needed to avoid keeping in memory
                if self.early_stopping_batches:
                    scores = []
                    start = 0
                    for i in range(1, self.n_batches+1):
                        end = (i * len(y_test_bin) // self.n_batches)
                        batch_idx = test_indices[start:end]
                        batch_idx_mod_n = batch_idx % n_test
                        if sparse.issparse(X_test):
                            X_test_batch = sparse.hstack((X_test[batch_idx_mod_n], thresholds[batch_idx // n_test]))
                        else:
                            X_test_batch = np.hstack((X_test[batch_idx_mod_n], thresholds[batch_idx // n_test]))
                        
                        scores.append(model.score(X_test_batch, y_test_bin[batch_idx]))
                        start = end          
                        
                    cur_score = np.mean(scores)
                
                else:
                    cur_score = model.score(X_test_bin, y_test_bin)

                if cur_score - self.tol > best_score:
                    best_score = cur_score
                    n_no_improvement = 0
                else:
                    n_no_improvement += 1
                    if n_no_improvement >= self.n_iter_no_change:
                        if self.verbosity >= 2:
                            print("Stopped early at epoch ", epoch, " Current score:", cur_score)
                        break

            if self.shuffle:
                sampled_indices = rng.integers(len(y_bin), size=len(y_bin))

        coef, intercept = tuple(model.parameters())
        coef = coef.detach().numpy().flatten()
        intercept = intercept.detach().numpy().flatten()
        self.coef_ = coef[:-self.k_]
        self.intercept_ = coef[-self.k_:] +  intercept
        return self
