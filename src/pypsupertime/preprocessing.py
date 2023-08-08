import warnings
import numpy as np
from typing import Union
from collections.abc import Iterable
import scanpy as sc
import anndata as ad
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import class_weight


Numeric = Union[int, float, np.number]


def restructure_y_to_bin(y_orig):
    """ 
    The given labels are converted to a binary representation,
    such that the threshold from 0-1 corresponds from changing from label
    $l_i$ to $l_{i+1}$. 
    $k$ copies of the label vector are concatenated such that for every
    vector $j$ the labels  $l_i$ with $i<j$ are converted to 0 and the 
    labels $i < j$ are converted to 1.

    :param y_orig: Original data set labels
    :type y_orig: Iterable
    :return: Restructured Labels: array of length n * k
    :rtype: numpy array
    """
    y_classes = np.unique(y_orig)
    k = len(y_classes)

    y_bin = []
    for ki in range(1,k):
        thresh = y_classes[ki]
        y_bin += [int(x >= thresh) for x in y_orig]

    y_bin = np.array(y_bin)

    return y_bin


def restructure_X_to_bin(X_orig, n_thresholds):
    """
    The count matrix is extended with copies of itself, to fit the converted label
    vector FOR NOW. For big problems, it could suffice to have just one label 
    vector and perform and iterative training.
    To train the thresholds, $k$ columns are added to the count matrix and 
    initialized to zero. Each column column represents the threshold for a 
    label $l_i$ and is set to 1, exactly  where that label $l_1$ occurs.

    :param X_orig: input data
    :type X_orig: numpy array with shape (n_cells, n_genes)
    :param n_thresholds: number of thresholds to be learned - equal to num_unique_labels - 1
    :type n_thresholds: integer
    :return: Restructured matrix of shape (n_cells * n_thresholds, n_genes + n_thresholds)
    :rtype: numpy array
    """

    # X training matrix
    X_bin = np.concatenate([X_orig.copy()] * (n_thresholds))
    # Add thresholds
    num_el = X_orig.shape[0] * (n_thresholds)

    for ki in range(n_thresholds):
        temp = np.repeat(0, num_el).reshape(X_orig.shape[0], (n_thresholds))
        temp[:,ki] = 1
        if ki > 0:
            thresholds = np.concatenate([thresholds, temp])
        else:
            thresholds = temp

    X_bin = np.concatenate([X_bin, thresholds], axis=1)

    return X_bin


def transform_labels(y: Iterable[Numeric]):
    """
    Transforms a target vector, such that it contains successive labels starting at 0.

    :param y: Iterable containing the ordinal labels of a dataset. Note: Must be number (int, float, np.number)!
    :type y: Iterable[number]
    :return: Numpy array with the labels converted
    :rtype: numpy.array
    """

    # convert to numeric 
    try:
        y = np.array(y).astype("float32")
    except ValueError as e:
        print(e)
        raise ValueError("Error Converting labels to numeric values")

    labels = np.unique(y)
    ordering = labels.argsort()
    y_trans = np.zeros_like(y)
    for i, el in enumerate(y):
        for l, o in zip(labels, ordering):
            if el == l:
                y_trans[i] = o

    return y_trans


def calculate_weights(y):
    """
    Calculates weights from the classes in y. 
    Returns an array the same length as y, 
    where the class is replaced with their respective weight

    Calculates balanced class weights according to
    `n_samples / (n_classes * np.bincount(y))`
    as is done in sklearn.
    """
    classes = np.unique(y)
    weights = class_weight.compute_class_weight("balanced", classes=classes, y=y)
    
    transf = dict(zip(classes, weights))

    return np.array([transf[e] for e in y])


def smooth(adata, knn=10, inplace=True):

    # corellate all cells
    cor_mat = np.corrcoef(adata.X)

    # calculate the ranks of cell correlations
    order_mat = np.argsort(cor_mat, axis=1)
    rank_mat = np.argsort(order_mat, axis=1)

    # indicate the knn closest neighbours
    idx_mat = rank_mat <= knn

    # calculate the neighborhood average
    avg_knn_mat = idx_mat / np.sum(idx_mat, axis=1, keepdims=True)
    assert np.all(np.sum(avg_knn_mat, axis=1) == 1)

    imputed_mat = np.dot(avg_knn_mat, adata.X)

    if not inplace:
        adata = adata.copy()

    adata.X = imputed_mat
    return adata


class Preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 log: bool = False,
                 scale: bool = True,
                 normalize: bool = False,
                 smooth: bool = False,
                 smooth_knn: int = 10,
                 select_genes: str = "all",
                 gene_list: Iterable | None = None,
                 min_gene_mean: float = 0.1,
                 max_gene_mean: float = 3,
                 hvg_min_dispersion: float = 0.5,
                 hvg_max_dispersion: float = np.inf,
                 hvg_n_top_genes: int | None = None):
        
        self.scale = scale
        self.log = log
        self.normalize = normalize
        self.smooth = smooth
        self.smooth_knn = smooth_knn
        self.min_gene_mean = min_gene_mean
        self.max_gene_mean = max_gene_mean
        self.hvg_min_dispersion = hvg_min_dispersion
        self.hvg_max_dispersion = hvg_max_dispersion
        self.hvg_n_top_genes = hvg_n_top_genes

        # Validate gene selection method
        select_genes_options = {"all", "hvg", "tf_mouse", "tf_human", "list"}
        self.select_genes = select_genes
        if self.select_genes not in select_genes_options:
            raise ValueError("Parameter select_genes must be one of %s." % select_genes_options)

        # Validate gene list
        self.gene_list = gene_list
        if self.gene_list is not None:
            if not isinstance(self.gene_list, Iterable):
                raise ValueError("Parameter gene_list must be an Iterable")
            
            if self.select_genes != "list":
                warnings.warn("Parameter select_genes was set to '%s' but gene_list was given and will be used" % self.select_genes)
                self.select_genes = "list"

            self.gene_list = np.array(self.gene_list)


    def fit_transform(self, adata: ad.AnnData, y: Iterable | None = None, **fit_params) -> ad.AnnData:
        
        # filter genes by their minimum mean counts
        cell_thresh = np.ceil(0.01 * adata.n_obs)
        sc.pp.filter_genes(adata, min_cells=cell_thresh)

        # log transform: adata.X = log(adata.X + 1)
        if self.log: sc.pp.log1p(adata, copy=False)

        # select highly-variable genes
        if self.select_genes == "hvg":
            sc.pp.highly_variable_genes(
                adata, flavor='seurat', n_top_genes=self.hvg_n_top_genes,
                min_disp=self.hvg_min_dispersion, max_disp=self.hvg_max_dispersion, inplace=True
            )
            adata = adata[:,adata.var.highly_variable].copy()
        
        # select mouse transcription factors
        elif self.select_genes == "tf_mouse":
            raise NotImplemented("select_genes='tf_mouse'")

        # select human transcription factors
        elif self.select_genes == "tf_human":
            raise NotImplemented("select_genes='tf_human'")

        # select curated genes from list
        # TODO: Check for array
        elif self.select_genes == "list":
            adata = adata[:, self.gene_list].copy()

        # select all genes
        elif self.select_genes == "all":
            pass

        # smoothing over neighbors to denoise data
        if self.smooth:
            raise NotImplementedError("Smoothing not working properly yet")
            adata = smooth(adata, knn=self.smooth_knn)

        # normalize with total UMI count per cell
        # this helps keep the parameters small
        if self.normalize: sc.pp.normalize_per_cell(adata)

        # scale to unit variance and shift to zero mean
        if self.scale: sc.pp.scale(adata)

        return adata
