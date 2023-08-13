from .preprocessing import Preprocessing, transform_labels
from .model import BatchSGDModel, BaselineSGDModel
from .parameter_search import RegularizationSearchCV

import datetime
import sys
import warnings
from typing import Iterable, Union

import numpy as np
from sklearn import metrics
import anndata as ad
from scanpy import read_h5ad

class Psupertime:

    def __init__(self,
                 max_memory=None,
                 n_folds=5,
                 n_jobs=5,
                 n_batches=1,
                 verbosity=1,
                 regularization_params=dict(),
                 preprocessing_params=dict(),
                 estimator_params=dict()):
        
        self.verbosity = verbosity
        
        # grid search params
        self.n_jobs = n_jobs
        self.n_folds = n_folds
        
        # model params
        self.max_memory = max_memory
        self.n_batches = n_batches
        
        if not isinstance(preprocessing_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: ", preprocessing_params)
        
        self.preprocessing = Preprocessing(**preprocessing_params)

        # Validate estimator params and instantiate model
        if not isinstance(estimator_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: %s" % estimator_params)
        
        self.estimator_params = estimator_params
        self.estimator_params["n_batches"] = self.n_batches
        self.model = None  # not fitted yet

        if not isinstance(regularization_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: ", regularization_params)
        
        _model_class = BatchSGDModel
        regularization_params["n_jobs"] = regularization_params.get("n_jobs", self.n_jobs)
        regularization_params["n_folds"] = regularization_params.get("n_folds", self.n_folds)
        regularization_params["estimator"] = _model_class
        self.grid_search = RegularizationSearchCV(**regularization_params)

    def run(self, adata: Union[ad.AnnData, str], ordinal_data: Union[Iterable, str]):
        
        start_time = datetime.datetime.now()

        # TODO: respect verbosity setting everywhere

        # Validate adata or load the filename
        if isinstance(adata, str):
            filename = adata
            adata = read_h5ad(filename)
        
        elif not isinstance(adata, ad.AnnData):
            raise ValueError("Parameter adata must be a filename or anndata.AnnData object. Received: ", adata)

        # Validate the ordinal data
        if isinstance(ordinal_data, str):
            column_name = ordinal_data
            if column_name not in adata.obs.columns:
                raise ValueError("Parameter ordinal_data is not a valid column in adata.obs. Received: ", ordinal_data)

            ordinal_data = adata.obs.get(column_name)
        
        elif isinstance(ordinal_data, Iterable):
            if len(ordinal_data) != adata.n_obs:
                raise ValueError("Parameter ordinal_data has invalid length. Expected: %s Received: %s" % (len(ordinal_data), len(adata.n_obs)))

        adata.obs["ordinal_label"] = transform_labels(ordinal_data)

        if self.max_memory is not None:
            # TODO: Validate number 
            #   -> is it int?
            #   -> Is it bigger than the object size?
            
            # TODO: Determine number of batches needed to keep memory usage below max_memory
            bytes_per_gb = 1000000000
            gb_size = sys.getsizeof(adata) / bytes_per_gb

            raise NotImplementedError("Max Memory cannot be set currently")

        # Run Preprocessing
        print("Preprocessing", end="\r")
        adata = self.preprocessing.fit_transform(adata)
        print("Preprocessing: done. n_genes=%s, n_cells=%s" % (adata.n_vars, adata.n_obs))

        # TODO: Test / Train split required? -> produce two index arrays, to avoid copying the data?

        # Run Grid Search
        print("Grid Search CV: CPUs=%s, n_folds=%s" % (self.grid_search.n_jobs, self.grid_search.n_folds))
        self.grid_search.fit(adata.X, adata.obs.ordinal_label, estimator_params=self.estimator_params)

        # Refit Model on _all_ data
        print("Refit on all data", end="\r")
        self.model = self.grid_search.get_optimal_model("1se")
        self.model.fit(adata.X, adata.obs.ordinal_label)
        acc = metrics.accuracy_score(self.model.predict(adata.X), adata.obs.ordinal_label)
        dof = np.count_nonzero(self.model.coef_)
        print("Refit on all data: done. accuracy=%f.02, n_genes=%s" % (acc, dof))

        # Annotate the data
        self.model.predict_psuper(adata, inplace=True)

        # TODO: Produce plots automatically?

        print("Total elapsed time: ", str(datetime.datetime.now() - start_time))
        
        # inplace
        # return adata
