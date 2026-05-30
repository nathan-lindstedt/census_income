import numpy as np
import pandas as pd
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def famd_fit(X_train: pd.DataFrame, X_hot_vars: List[str]) -> tuple:
    """Fit FAMD (Factor Analysis of Mixed Data) on training data.

    Normalises OHE binary columns using the Greenacre method — centering
    each column by its proportion p_j and scaling by the binomial standard
    deviation sqrt(p_j * (1 - p_j)) — then standardises continuous columns
    as in PCA, and applies SVD to the combined normalised matrix. Returns
    the first component scores for the training set together with all fitted
    objects needed to project new data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix with OHE binary and continuous columns.
    X_hot_vars : List[str]
        Original categorical variable names before OHE, used to identify
        which columns in X_train are OHE-encoded binary indicators.

    Returns
    -------
    lens : (n_train, 1) ndarray
        First FAMD component scores for the training set.
    ohe_cols : List[str]
        OHE column names identified in X_train.
    cont_cols : List[str]
        Continuous column names in X_train.
    p_j : (n_ohe,) ndarray
        Column proportions used for Greenacre normalisation.
    scaler : StandardScaler
        Fitted scaler for continuous columns.
    pca : PCA
        Fitted PCA(n_components=1) whose loadings define the FAMD axis.
    """
    ohe_cols = [col for col in X_train.columns
                if any(col.startswith(label + '_') for label in X_hot_vars)]
    cont_cols = [col for col in X_train.columns if col not in ohe_cols]

    Z_train = X_train[ohe_cols].values
    p_j = np.clip(Z_train.mean(axis=0), 1e-10, 1 - 1e-10)
    Z_norm = (Z_train - p_j) / np.sqrt(p_j * (1 - p_j))

    scaler = StandardScaler()
    C_norm = scaler.fit_transform(X_train[cont_cols].values)

    pca = PCA(n_components=1)
    lens = pca.fit_transform(np.hstack([Z_norm, C_norm]))

    return lens, ohe_cols, cont_cols, p_j, scaler, pca


def famd_transform(X_new: pd.DataFrame, ohe_cols: List[str], cont_cols: List[str],
                   p_j: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    """Project new data onto a fitted FAMD axis.

    Applies the same Greenacre normalisation (using the training proportions
    p_j) and StandardScaler fitted by famd_fit, then projects through the
    stored PCA loadings. This places new observations in the same factor
    space as the training set without refitting the decomposition.

    Parameters
    ----------
    X_new : pd.DataFrame
        Feature matrix with the same OHE and continuous columns as the
        training data used in famd_fit.
    ohe_cols : List[str]
        OHE column names (returned by famd_fit).
    cont_cols : List[str]
        Continuous column names (returned by famd_fit).
    p_j : (n_ohe,) ndarray
        Column proportions for Greenacre normalisation (returned by famd_fit).
    scaler : StandardScaler
        Fitted scaler for continuous columns (returned by famd_fit).
    pca : PCA
        Fitted PCA(n_components=1) (returned by famd_fit).

    Returns
    -------
    lens : (n_new, 1) ndarray
        First FAMD component scores for the new observations.
    """
    Z_new = X_new[ohe_cols].values
    Z_norm = (Z_new - p_j) / np.sqrt(p_j * (1 - p_j))
    C_norm = scaler.transform(X_new[cont_cols].values)
    return pca.transform(np.hstack([Z_norm, C_norm]))
