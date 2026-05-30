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
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

from census_income import helper_funcs

#%%
# System path variables
# Resolves the directory of this script file so that relative paths used for
# loading and saving model artifacts are anchored to the file location rather
# than the working directory of the calling process.
start: str = os.path.dirname(__file__)

#%%
# Global variable initialization
# These lists are populated by helper_funcs.feature_names after transformation
# and are later passed to KeplerMapper for labeling nodes in the graph
# visualizations.
X_feat_names: List[str] = []
y_feat_names: List[str] = []

#%%
# XGBoost hyperparameter initialization
# Fixed hyperparameters that are held constant across the grid search.
# min_child_weight regularizes the tree by requiring a minimum sum of instance
# weights in a leaf. max_bin controls histogram bin granularity for the 'hist'
# tree method. num_parallel_tree sets the number of trees per boosting round,
# effectively making this a random forest ensemble inside XGBoost. subsample,
# colsample_bytree, and colsample_bynode inject stochasticity at the sample,
# tree, and node levels respectively to reduce overfitting.
min_child_weight: int = 8
max_bin: int = 48
num_parallel_tree: int = 64
subsample: float = 0.8
colsample_bytree: float = 0.8
colsample_bynode: float = 0.8
verbose: bool = True

#%%
# Load the dataset
# Fetches the UCI Adult (Census Income) dataset via its repository ID (20).
# This dataset contains US Census demographic data with the task of predicting
# whether an individual's annual income exceeds $50,000.
adult = fetch_ucirepo(id=20) 

#%%
# Features and target variable
# Separates the fetched dataset into the feature matrix and target vector.
# Both are kept as raw pandas DataFrames at this stage so that original dtypes
# (object for categoricals, int/float for numerics) are preserved for
# downstream preprocessing and for computing the Gower distance matrix later.
X_import = adult.data.features 
y_import = adult.data.targets

#%%
# Data cleaning
# The UCI source encodes some income labels with a trailing period
# (e.g., ">50K." vs ">50K"). Stripping the period normalizes the two
# formatting variants to a consistent set of labels before encoding.
y_import = pd.DataFrame(y_import['income'].str.replace('.',''))

#%%
# X variable one-hot encoding
# Identifies all object-dtype (categorical) columns and applies
# OneHotEncoder with drop='first' to avoid perfect multicollinearity
# (the dummy variable trap). sparse_output=False returns a dense array
# compatible with downstream NumPy and scikit-learn operations. Numeric
# columns are passed through unchanged via remainder='passthrough'.
X_hot_vars = X_import.select_dtypes(include='object').columns.tolist()

X_prep = make_column_transformer(
    (OneHotEncoder(drop='first', sparse_output=False), X_hot_vars),
    remainder='passthrough'
)

X = X_prep.fit_transform(X_import)

#%%
# One-hot encoding feature names
# Reconstructs human-readable column names from the fitted ColumnTransformer
# and wraps the transformed array in a pandas DataFrame. Preserving the
# original integer index is critical so that rows can later be aligned back to
# X_import by index when computing the Gower distance matrix on pre-OHE data.
X = helper_funcs.feature_names(X_feat_names, X_prep, X)

#%%
# Y variable one-hot encoding
# Encodes the binary income label (<=50K / >50K) as a 0/1 numeric column.
# drop='first' makes <=50K the reference class (0) and >50K the positive
# class (1), which aligns with the convention used throughout model evaluation
# and the KeplerMapper miss visualization.
y_prep = OneHotEncoder(drop='first', sparse_output=False)
y_feat_names = y_import.columns.tolist()
y = pd.DataFrame(y_prep.fit_transform(y_import[y_feat_names].values.reshape(-1,1)))

#%%
# Train, test, validation split
# Produces an approximate 60/20/20 train/validation/test split via two
# sequential calls. The first holds out 20% as the final test set. The second
# splits the remaining 80% into 75/25, yielding 60% train and 20% validation.
# A fixed random_state ensures reproducibility. The validation set is used for
# early stopping during final model training and for reporting held-out metrics.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#%%
# XGBoost Random Forest class weight
# The Census Income dataset is imbalanced: roughly 75% of observations earn
# <=50K. scale_pos_weight compensates by up-weighting the minority class
# (>50K) during training. The standard formula is the ratio of negative to
# positive class counts in the training set.
count_0 = int(len(y_train[y_train == 0]))
count_1 = int(len(y_train[y_train == 1]))
xgbrf_class_weight = float(count_0 / count_1)

#%%
# XGBoost Random Forest tuning and training
# A trained model is cached to disk (keyed by sklearn version) and reloaded on
# subsequent runs to avoid retraining. If no cached model exists,
# HalvingGridSearchCV performs successive-halving search over max_depth, gamma,
# and learning_rate, using n_estimators as the resource that is tripled each
# round (factor=3) from min_resources=2 up to max_resources=500. The best
# parameters from the search are then used to train a final model with early
# stopping on the validation set to find the optimal number of trees.
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
# Evaluates the final model on training and validation sets. Precision and
# recall capture class-level performance on the imbalanced positive class
# (>50K). ROC AUC summarizes discrimination across all probability thresholds
# and is the primary quality metric, matching the scoring used during search.
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
# KeplerMapper implements the Mapper algorithm from Topological Data Analysis
# (TDA). It constructs a graph summarizing the shape of the data by covering
# a low-dimensional lens space with overlapping intervals, clustering points
# within each interval under a chosen metric, and connecting clusters that
# share points across adjacent intervals.
mapper = km.KeplerMapper(verbose=1)

#%%
# Gower distance matrix
# Compute on original pre-OHE data with explicit categorical feature flags so
# that categorical columns use simple matching rather than being treated as
# continuous binary variables after one-hot encoding.
X_hot_vars_mask = np.array([col in X_hot_vars for col in X_import.columns])
X_import_train = X_import.loc[X_train.index]
X_train_gower = gower.gower_matrix(X_import_train, cat_features=X_hot_vars_mask)

#%%
# Create 2-D lens with XGBoost Random Forest and Spectral Embedding
# SpectralEmbedding (Laplacian Eigenmaps) is derived directly from the Gower
# affinity matrix, making the lens geometrically consistent with the
# precomputed Gower distances used for clustering.
lens_1 = xgbrf_model.predict_proba(X_train)[:,1].reshape((X_train.shape[0], 1))
sigma = np.median(X_train_gower)
affinity = np.exp(-X_train_gower**2 / (2 * sigma**2))
lens_2 = SpectralEmbedding(n_components=1, affinity='precomputed', random_state=1).fit_transform(affinity)
lenses = np.c_[lens_1, lens_2]

#%%
# Create the Kepler Mapper graph
# Cover divides each lens dimension into n_cubes=20 overlapping intervals with
# 10% overlap, controlling graph resolution and connectivity. Agglomerative
# clustering with average linkage is applied within each cover element using
# the precomputed Gower distance matrix; n_clusters=2 reflects the binary
# income classification task. precomputed=True tells Mapper that the second
# argument is a pairwise distance matrix, not raw feature data.
graph = mapper.map(
    lenses,
    X_train_gower,
    cover=km.Cover(n_cubes=20, perc_overlap=.10),
    clusterer=AgglomerativeClustering(metric='precomputed', linkage='average', n_clusters=2),
    precomputed=True
)

#%%
# Visualize the Kepler Mapper graph by target variable
# Renders the Mapper graph as an interactive HTML file. custom_tooltips
# displays the true income label for each observation when hovering over a
# node. Nodes are colored by mean and median of both lenses, providing a
# simultaneous supervised (XGBoost probability) and unsupervised (Spectral
# Embedding coordinate) view of each graph node's composition.
mapper.visualize(
    graph,
    path_html="../../census-income-xgb-train-targets.html",
    title="Census Income",
    custom_tooltips=np.array(y_train[0]),
    color_values=lenses,
    X=np.array(X_train),
    X_names=X_feat_names,
    color_function_name=["XGBoost Random Forest", "Spectral Embedding"],
    node_color_function=["mean", "median"]
)

#%%
# Visualize the Kepler Mapper graph by misses
# A second visualization colors nodes by model error. custom_tooltips show the
# absolute difference between the true label and the predicted label for each
# observation, making it straightforward to identify topological regions of
# the graph where the model systematically misclassifies.
mapper.visualize(
    graph,
    path_html="../../census-income-xgb-train-misses.html",
    title="Census Income",
    custom_tooltips=np.array(abs(y_train[0] - xgbrf_model.predict(X_train))),
    color_values=lenses,
    X=np.array(X_train),
    X_names=X_feat_names,
    color_function_name=["XGBoost Random Forest", "Spectral Embedding"],
    node_color_function=["mean", "median"]
)

#%%
