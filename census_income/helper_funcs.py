import numpy as np
import pandas as pd
from numpy.linalg import inv, solve
from numpy import matmul as mm
from typing import List
from sklearn.compose import ColumnTransformer

def feature_names(X_names_list: List[str], X_transformer: ColumnTransformer, X: np.ndarray) -> pd.DataFrame:
    """
    Generates a DataFrame with feature names after transformation.
    This function takes a list of feature names, a ColumnTransformer object, 
    and a numpy array of transformed data, and returns a pandas DataFrame 
    with the appropriate feature names.

    Parameters
    ----------
    X_names_list : List[str] 
        A list to which the feature names will be appended.
    X_transformer : ColumnTransformer
        A fitted ColumnTransformer object that contains the transformations applied to the data.
    X : np.ndarray 
        A numpy array containing the transformed data.
    
    Returns
    ----------
    X : pd.DataFrame
        A pandas DataFrame with the transformed data and the corresponding feature names.
    """
    for name, transformer, features, _ in X_transformer._iter(
        fitted=True, column_as_labels=True, skip_drop=True, skip_empty_columns=True
    ):
        if transformer != 'passthrough':
            try:
                X_names_list.extend(
                    X_transformer.named_transformers_[name].get_feature_names_out()
                )
            except AttributeError:
                X_names_list.extend(features)
        else:
            X_names_list.extend(X_transformer._feature_names_in[features])

    return pd.DataFrame(X, columns=X_names_list)


def logistic_pca(X: np.ndarray, num_components: int=None, num_iter: int=50, lambda_l1: float=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logistic principal component analysis (PCA) w/ optional
    L1 regularization for inducing sparsity (experimental).

    Parameters
    ----------
    X : (num_samples, num_dimensions) ndarray
        Data matrix.
    num_components : int, optional
        Number of PCA components.
    num_iter : int, default=50
        Number iterations for fitting model.
    lambda_l1 : float, optional
        L1 regularization parameter (experimental).

    Returns
    ----------
    W : (num_dimensions, num_components) ndarray
        Estimated projection matrix.
    mu : (num_components, num_samples) ndarray
        Estimated latent variables.
    b : (num_dimensions, 1) ndarray
        Estimated bias.

    References
    ----------
    Tipping, Michael E. "Probabilistic visualisation of high-dimensional binary data." 
    Advances in neural information processing systems (1999): 592-598.

    Lee, Seokho, Jianhua Z. Huang, and Jianhua Hu. "Sparse logistic principal components analysis for binary data."
    The Annals of Applied Statistics 4.3 (2010): 1579-1601.

    Copyright (c) 2021 Mikael Brudfors under MIT License
    """
    num_samples: int = X.shape[0]
    num_dimensions: int = X.shape[1]
    num_components: int = _get_num_components(num_components, num_samples, num_dimensions)
    
    # Constants
    N: int = num_samples
    D: int = num_dimensions
    K: int = num_components
    
    # Initialize variables
    # I: Identity matrix
    I: np.ndarray = np.eye(K)
    # W: Projection matrix
    W: np.ndarray = np.random.randn(D, K)
    # mu: Latent variables
    mu: np.ndarray = np.random.randn(K, N)
    # b: Bias
    b: np.ndarray = np.random.randn(D, 1)
    # C: Covariance matrix
    C: np.ndarray = np.repeat(I[:, :, np.newaxis], N, axis=2)
    # xi: Variational parameters
    xi: np.ndarray = np.ones((N, D))
    
    # Functions
    # Sigmoid and lambda function
    sig = lambda x: 1/(1 + np.exp(-x))
    lam = lambda x: (0.5 - sig(x))/(2*x)
    
    # Fit model
    for iter in range(num_iter):
        # Step 1. Obtain the sufficient statistics for the approximated posterior 
        # distribution of latent variables given each observation
        for n in range(N):
            # Get sample
            x_n = X[n, :][:, None]
           
            # Compute approximation
            lam_n = lam(xi[n, :])[:, None]
            
            # Update covariance matrix and latent variables
            C[:, :, n] = inv(I - 2*mm(W.T, lam_n*W))
            mu[:, n] = mm(C[:, :, n], mm(W.T, x_n - 0.5 + 2*lam_n*b))[:, 0]
        
        # Step 2. Optimise the variational parameters in in order to make the 
        # approximation as close as possible
        for n in range(N):
            # Posterior statistics
            z = mu[:, n][:, None]
            E_zz = C[:, :, n] + mm(z, z.T)
            
            # Variational parameters xi squared
            xixi = np.sum(W*mm(W, E_zz), axis=1, keepdims=True) \
                   + 2*b*mm(W, z) + b**2

            # Update variational parameters
            xi[n, :] = np.sqrt(np.abs(xixi[:, 0]))
        
        # Step 3. Update model parameters
        E_zhzh = np.zeros((K + 1, K + 1, N))

        for n in range(N):
            z = mu[:, n][:, None]
            E_zhzh[:-1, :-1, n] = C[:, :, n] + mm(z, z.T)
            E_zhzh[:-1, -1, n] = z[:, 0]
            E_zhzh[-1, :-1, n] = z[:, 0]
            E_zhzh[-1, -1, n] = 1
        
        E_zh = np.append(mu, np.ones((1, N)), axis=0)
        
        for i in range(D):
            # Compute approximation
            lam_i = lam(xi[:, i])[None][None]
            
            # Hessian and gradient
            if lambda_l1 is None:
                H = np.sum(2*lam_i*E_zhzh, axis=2)
                g = mm(E_zh, X[:, i] - 0.5) 
            
            # Hessian and gradient w/ L1 regularization (experimental)
            else:
                H = np.sum(2*lam_i*E_zhzh, axis=2)
                g = mm(E_zh, X[:, i] - 0.5) - lambda_l1 * np.sign(np.append(W[i, :], b[i]))
            
            # Invert Hessian
            wh_i = -solve(H, g[:, None])
            wh_i = wh_i[:, 0]
            
            # Update the projection matrix and bias
            W[i, :] = wh_i[:K]
            b[i] = wh_i[K]

    return W, mu, b


def pca(X: np.ndarray, num_components: int=None, zero_mean: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """Principal component analysis (PCA).

    Parameters
    ----------
    X : (num_samples, num_dimensions) ndarray
        Data matrix.
    num_components : int, optional
        Number of PCA components.
    zero_mean : bool, default=True
        Zero mean data.

    Returns
    ----------
    W : (num_dimensions, num_components) ndarray
        Principal axes.        
    mu : (num_components, ) ndarray
        Principal components.    

    Copyright (c) 2021 Mikael Brudfors under MIT License
    """
    num_samples: int = X.shape[0]
    num_dimensions: int = X.shape[1]
    num_components: int = _get_num_components(num_components, num_samples, num_dimensions)
    
    # Zero mean data
    if zero_mean:     
        X -= np.mean(X, axis=0)  
    
    # Compute covariance matrix
    X = np.cov(X, rowvar=False)
    
    # Eigen decomposition
    mu: np.ndarray
    W: np.ndarray
    mu, W = np.linalg.eig(X)
    
    # Sort descending order
    idx: np.ndarray = np.argsort(mu)[::-1]
    W = W[:, idx]
    mu = mu[idx]
    
    # Extract components
    mu = mu[:num_components]
    W = W[:, :num_components]

    return W, mu
    
    
def _get_num_components(num_components: int, num_samples: int, num_dimensions: int) -> int:
    """Get number of components (clusters).

    Parameters
    ----------
    num_components : int
        Number of PCA components.
    num_samples : int
        Number of samples in the dataset.
    num_dimensions : int
        Number of dimensions in the dataset.

    Returns
    ----------
    num_components : int
        Number of components to use.
    """
    if num_components is None:
        num_components = min(num_samples, num_dimensions)    

    return num_components
