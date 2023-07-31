import numpy as np
from collections.abc import Iterable
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
import warnings


class RegularizationSearchCV:

    def __init__(self, 
                 estimator: BaseEstimator, 
                 scoring=None, 
                 reg_param_name="regularization", 
                 reg_path=None, 
                 n_params=40, 
                 reg_high=1, 
                 reg_low=0.001, 
                 n_jobs=-1, 
                 n_folds=5):
        
        self.is_fitted = False
        self.n_jobs = n_jobs
        self.n_folds = n_folds

        self.scoring = scoring  # for now, only use default scoring
        if not isinstance(scoring, str) and isinstance(scoring, dict) or isinstance(scoring, Iterable):
            warnings.warn("Parameter 'scoring' is a list or dict: Multiple scorers are currently not supported. Using the model default.")
            self.scoring = None

        if reg_path is None:
            self.reg_path = np.geomspace(reg_high, reg_low, n_params)
        else:
            try:
                self.reg_path = np.array(reg_path).astype("float")
            except ValueError as e:
                print(e)
                raise ValueError("Parameter 'reg_path' is not Iterable or cannot be converted to float")

        self.estimator = estimator
        if not isinstance(self.estimator, BaseEstimator):
            raise ValueError("Parameter 'estimator' is not a sklern.base.BaseEstimator")

        self.reg_param_name = reg_param_name
        try:
            _ = self.estimator().get_params()[self.reg_param_name]
        except KeyError as e:
            print(e)
            raise ValueError("Parameter 'reg_param_name' is not a valid parameter for %s" % self.estimator.__class__) 

        # average cross validation scores for each lambda
        self.scores = []

        # std of cross validation scores for each lambda
        self.scores_std = []

        # average training scores fore each lambda
        self.train_scores = []

        # std of training scores fore each lambda
        self.train_scores_std = []

        # best degrees of freedom 
        self.dof = []

        # best estimators
        self.fitted_estimators = []

    def fit(self, X, y, fit_params=dict(), estimator_params=dict()):

        for i, reg in enumerate(self.reg_path):
            
            print("Regularization: %s/%s" % (i+1, len(self.reg_path)), sep="", end="\r")

            estimator_params[self.reg_param_name] = reg
            cv = cross_validate(estimator=self.estimator(**estimator_params),
                                scoring=self.scoring,
                                n_jobs=self.n_jobs,
                                cv=self.n_folds,
                                X=X,
                                y=y,
                                error_score="raise",
                                return_train_score=True,
                                return_estimator=True,
                                fit_params=fit_params
                                )

            best_idx = np.argmax(cv["test_score"])
            self.train_scores.append(np.mean(cv["train_score"]))
            self.train_scores_std.append(np.std(cv["train_score"]))
            self.scores.append(np.mean(cv["test_score"]))
            self.scores_std.append(np.std(cv["test_score"]))
            self.fitted_estimators.append(cv["estimator"][best_idx])

            # TODO: disregard the thresholds when fitting the binary logistic model
            self.dof.append(np.count_nonzero(np.array(cv["estimator"][best_idx].coef_).flatten()))

        print("Regularization: done")
        self.is_fitted_ = True
        return self

    def get_optimal_lambda(self, method="1se", index=None):

        if not method in ["1se", "best", "index"]:
            raise ValueError("The method parameter should be one of '1se' or 'best'")

        if method =="index":
            if index is None or (index >= len(self.scores) or index < 0): 
                raise ValueError("Parmeter `index` must be set to a valid cv index, if method='index' is selected.")
            return (self.lambdas[index], index)

        if method == "best":
            idx = np.argmax(self.scores)
            return (self.lambdas[idx], idx)
            
        if method == "1se":
            n = len(self.dof)

            # check the effect direction of the regularization parameter
            sparsity_increases_w_idx = np.mean(self.dof[:n//4]) < np.mean(self.dof[-n//4:])

            # compute the threshold as the maximum score minus the standard error
            nonzero_idx = np.nonzero(self.dof)
            max_idx = np.argmax(self.scores)
            thresh = self.scores[max_idx] - np.std(np.array(self.scores)[nonzero_idx])

            if sparsity_increases_w_idx:
                items = zip(self.scores, self.dof)
            else:
                items = reversed(list(zip(self.scores, self.dof)))

            for i, (s, d) in enumerate(items):
                # exclude models with 0 degrees of freedom
                # and stop if there is no sufficiently good sparser model
                if (s > thresh and d != 0) or \
                   (i == max_idx):
                    return (self.lambdas[i], i)
            
            print("Warning: No model for method '1se' with non-zero degrees of freedom could be found. Returning the best scoring model")
            return self.get_optimal_lambda(method="best")
        
    def get_optimal_parameters(self, *args, **kwargs):
        lamb, idx = self.get_optimal_lambda(*args, **kwargs)
        return self.fitted_estimators[idx].get_params()

    def get_optimal_model(self, *args, **kwargs):
        return self.estimator(**self.get_optimal_parameters(*args, **kwargs))

    def get_estimator_weights_by_lambdas(self):
        pass
