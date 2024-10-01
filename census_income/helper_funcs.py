import numpy as np
import pandas as pd
from numpy.linalg import (inv, solve)
from numpy import matmul as mm

def feature_names(X_names_list, X_transformer, X):
    for name, transformer, features, _ in X_transformer._iter(fitted=True, column_as_labels=True, skip_drop=True, skip_empty_columns=True):
        if transformer != 'passthrough':
            try:
                X_names_list.extend(X_transformer.named_transformers_[name].get_feature_names_out())
            except AttributeError:
                X_names_list.extend(features)
            
        if transformer == 'passthrough':
            X_names_list.extend(X_transformer._feature_names_in[features])

    return pd.DataFrame(X, columns=X_names_list)


def logistic_pca(X, num_components=None, num_iter=50):
    """Logistic principal component analysis (PCA).

    Parameters
    ----------
    X : (num_samples, num_dimensions) ndarray
        Data matrix.
    num_components : int, optional
        Number of PCA components.
    num_iter : int, default=32
        Number iterations for fitting model.

    Returns
    ----------
    W : (num_dimensions, num_components) ndarray
        Estimated projection matrix.
    mu : (num_components, num_samples) ndarray
        Estimated latent variables.
    b : (num_dimensions, 1) ndarray
        Estimated bias.

    Reference
    ----------
    Tipping, Michael E. "Probabilistic visualisation of high-dimensional binary data." 
    Advances in neural information processing systems (1999): 592-598.

    """
    num_samples = X.shape[0]
    num_dimensions = X.shape[1]
    num_components = _get_num_components(num_components, num_samples, num_dimensions)
    
    # Variables
    N = num_samples
    D = num_dimensions
    K = num_components
    
    # Initialize variables
    I = np.eye(K)
    W = np.random.randn(D, K)
    mu = np.random.randn(K, N)
    b = np.random.randn(D, 1)    
    C = np.repeat(I[:, :, np.newaxis], N, axis=2)
    xi = np.ones((N, D))  # the variational parameters
    
    # Functions
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
            
            # Update
            C[:, :, n] = inv(I - 2*mm(W.T, lam_n*W))
            mu[:, n] = mm(C[:, :, n], mm(W.T, x_n - 0.5 + 2*lam_n*b))[:, 0]
        
        # Step 2. Optimise the variational parameters in in order to make the 
        # approximation as close as possible
        for n in range(N):
            
            # Posterior statistics
            z = mu[:, n][:, None]
            E_zz = C[:, :, n] + mm(z, z.T)
            
            # Xi squared
            xixi = np.sum(W*mm(W, E_zz), axis=1, keepdims=True) \
                   + 2*b*mm(W, z) + b**2
            # Update
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
            
            # Gradient and Hessian
            H = np.sum(2*lam_i*E_zhzh, axis=2)
            g = mm(E_zh, X[:, i] - 0.5)
            
            # Invert
            wh_i = -solve(H, g[:, None])
            wh_i = wh_i[:, 0]
            
            # Update
            W[i, :] = wh_i[:K]
            b[i] = wh_i[K]

    return W, mu, b


def pca(X, num_components=None, zero_mean=True):
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

    """
    num_samples = X.shape[0]
    num_dimensions = X.shape[1]
    num_components = _get_num_components(num_components, num_samples, num_dimensions)
    if zero_mean:        
        # Zero mean
        X -= np.mean(X, axis=0)  
    
    # Compute covariance matrix
    X = np.cov(X, rowvar=False)
    
    # Eigen decomposition
    mu, W = np.linalg.eig(X)
    
    # Sort descending order
    idx = np.argsort(mu)[::-1]
    W = W[:,idx]
    mu = mu[idx]
    
    # Extract components
    mu = mu[:num_components]
    W = W[:, :num_components]

    return W, mu
    
    
def _get_num_components(num_components, num_samples, num_dimensions):
    """Get number of components (clusters).
    """
    if num_components is None:
        num_components = min(num_samples, num_dimensions)    

    return num_components
