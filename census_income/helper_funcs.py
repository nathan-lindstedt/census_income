import numpy as np
import pandas as pd
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
