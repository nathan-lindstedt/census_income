#%%
# Import libraries
import joblib
import os
import time
from typing import List

import kmapper as km
import numpy as np
import pandas as pd
import gower
import sklearn

from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

from census_income import helper_funcs

#%%
# System path variables
start: str = os.path.dirname(__file__)

#%%
# Global variable initialization
X_feat_names: List[str] = []
y_feat_names: List[str] = []

#%%
# XGBoost hyperparameter initialization
min_child_weight: int = 8
max_bin: int = 48
num_parallel_tree: int = 64
subsample: float = 0.8
colsample_bytree: float = 0.8
colsample_bynode: float = 0.8
verbose: bool = True

#%%
# Load the dataset
adult = fetch_ucirepo(id=20) 

#%%
# Features and target variable
X_import = adult.data.features 
y_import = adult.data.targets

#%%
# Data cleaning
y_import = pd.DataFrame(y_import['income'].str.replace('.',''))

#%%
# X variable one-hot encoding
X_hot_vars = X_import.select_dtypes(include='object').columns.tolist()

X_prep = make_column_transformer(
    (OneHotEncoder(drop='first', sparse_output=False), X_hot_vars),
    remainder='passthrough'
)

X = X_prep.fit_transform(X_import)

#%%
# One-hot encoding feature names
X = helper_funcs.feature_names(X_feat_names, X_prep, X)

#%%
# Y variable one-hot encoding
y_prep = OneHotEncoder(drop='first', sparse_output=False)
y_feat_names = y_import.columns.tolist()
y = pd.DataFrame(y_prep.fit_transform(y_import[y_feat_names].values.reshape(-1,1)))

#%%
# Train, test, validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#%%
# XGBoost Random Forest class weight
count_0 = int(len(y_train[y_train == 0]))
count_1 = int(len(y_train[y_train == 1]))
xgbrf_class_weight = float(count_0 / count_1)

#%%
# XGBoost Random Forest tuning and training
if not os.path.isfile(os.path.relpath(f'../../census_income_xgb_model_v{sklearn.__version__}.pkl', start=start)):
    xgbrf_hyperparameters = [
        {
            'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
            'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
            'learning_rate': [0.01, 0.5, 1.0]
        }
    ]

    xgbrf_start = time.perf_counter()

    xgbrf_gridsearch = HalvingGridSearchCV(
        XGBClassifier(
            tree_method='hist',
            grow_policy='depthwise',
            min_child_weight=min_child_weight,
            max_bin=max_bin,
            num_parallel_tree=num_parallel_tree,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            scale_pos_weight=xgbrf_class_weight,
            eval_metric='logloss'
        ),
        xgbrf_hyperparameters,
        resource='n_estimators',
        factor=3,
        min_resources=2,
        max_resources=500,
        scoring='roc_auc',
        cv=3,
        aggressive_elimination=True,
        verbose=verbose
    )

    xgbrf_best_model = xgbrf_gridsearch.fit(
        np.array(X_train), 
        np.array(y_train), 
        eval_set=[(np.array(X_val), np.array(y_val))], 
        verbose=False
    )

    xgbrf_stop = time.perf_counter()

    xgbrf_model = XGBClassifier(
        tree_method='hist',
        grow_policy='depthwise',
        early_stopping_rounds=20,
        min_child_weight=min_child_weight,
        max_bin=max_bin,
        num_parallel_tree=num_parallel_tree,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bynode=colsample_bynode,
        scale_pos_weight=xgbrf_class_weight,
        eval_metric='logloss',
        **xgbrf_gridsearch.best_params_
    ).fit(
        np.array(X_train), 
        np.array(y_train), 
        eval_set=[(np.array(X_val), np.array(y_val))]
    )

    print(f'XGBoost Random Forest model trained in {(xgbrf_stop - xgbrf_start) / 60:.1f} minutes')
    print(f'Best XGBoost Random Forest parameters: {xgbrf_gridsearch.best_params_}')

    joblib.dump(xgbrf_model, os.path.relpath(f'../../census_income_xgb_model_v{sklearn.__version__}.pkl', start=start))

else:
    xgbrf_model = joblib.load(os.path.relpath(f'../../census_income_xgb_model_v{sklearn.__version__}.pkl', start=start))

#%%
# XGBoost Random Forest metrics
xgbrf_train_probs = xgbrf_model.predict_proba(X_train)[:, 1]
xgbrf_train_prec = precision_score(y_train, xgbrf_model.predict(X_train))
xgbrf_train_recall = recall_score(y_train, xgbrf_model.predict(X_train))
xgbrf_train_auc = roc_auc_score(y_train, xgbrf_train_probs)

xgbrf_val_probs = xgbrf_model.predict_proba(X_val)[:, 1]
xgbrf_val_prec = precision_score(y_val, xgbrf_model.predict(X_val))
xgbrf_val_recall = recall_score(y_val, xgbrf_model.predict(X_val))
xgbrf_val_auc = roc_auc_score(y_val, xgbrf_val_probs)

print(f'Overall accuracy for XGBoost Random Forest model (training): '
    f'{xgbrf_model.score(X_train, y_train):.4f}')
print(f'Overall precision for XGBoost Random Forest model (training): '
    f'{xgbrf_train_prec:.4f}')
print(f'Overall recall for XGBoost Random Forest model (training): '
    f'{xgbrf_train_recall:.4f}')
print(f'ROC AUC for XGBoost Random Forest model (training): '
    f'{xgbrf_train_auc:.4f}\n')
print(f'Overall accuracy for XGBoost Random Forest model (validation): '
    f'{xgbrf_model.score(X_val, y_val):.4f}')
print(f'Overall precision for XGBoost Random Forest model (validation): '
    f'{xgbrf_val_prec:.4f}')
print(f'Overall recall for XGBoost Random Forest model (validation): '
    f'{xgbrf_val_recall:.4f}')
print(f'ROC AUC for XGBoost Random Forest model (validation): '
    f'{xgbrf_val_auc:.4f}\n')

#%%
# Initialize KeplerMapper
mapper = km.KeplerMapper(verbose=1)

#%%
# Train Gower distance matrix
X_train_gower = gower.gower_matrix(X_train)

#%%
# Create 2-D lens with XGBoost Random Forest and L2-norm
lens_1 = xgbrf_model.predict_proba(X_train)[:,1].reshape((X_train.shape[0], 1))
lens_2 = mapper.fit_transform(X_train, projection='l2norm')
lenses = np.c_[lens_1, lens_2]

#%%
# Create the Kepler Mapper graph
graph = mapper.map(
    lenses,
    X_train_gower,
    cover=km.Cover(n_cubes=20, perc_overlap=.10),
    clusterer=AgglomerativeClustering(metric='precomputed', linkage='average', n_clusters=2),
    precomputed=True
)

#%%
# Visualize the Kepler Mapper graph by target variable
mapper.visualize(
    graph,
    path_html="census-income-xgb-train-targets.html",
    title="Census Income",
    custom_tooltips=np.array(y_train[0]),
    color_values=lenses,
    X=np.array(X_train),
    X_names=X_feat_names,
    color_function_name=["XGBoost Random Forest", "L2-norm"],
    node_color_function=["mean", "median"]
)

#%%
# Visualize the Kepler Mapper graph by misses
mapper.visualize(
    graph,
    path_html="census-income-xgb-train-misses.html",
    title="Census Income",
    custom_tooltips=np.array(abs(y_train[0] - xgbrf_model.predict(X_train))),
    color_values=lenses,
    X=np.array(X_train),
    X_names=X_feat_names,
    color_function_name=["XGBoost Random Forest", "L2-norm"],
    node_color_function=["mean", "median"]
)

#%%
