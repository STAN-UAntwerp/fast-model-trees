import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.base import BaseEstimator
from .cpilot import PILOT
from .constants import DEFAULT_DF_SETTINGS
from functools import partial


class PILOTWrapper(PILOT):
    def __init__(
        self,
        feature_idx: list[int] | np.ndarray,
        df_settings=None,
        min_sample_leaf=5,
        min_sample_alpha=5,
        min_sample_fit=5,
        max_depth=20,
        max_model_depth=100,
        max_features=-1,  # -1 means that it must be set
        max_pivot=None,  # None means no approximation, otherwise interpreted as max pivots per feature
        rel_tolerance=0.01,
        precision_scale=1e-10,
    ):
        if max_features == -1:
            raise ValueError("max_features must be set")

        # Use default df_settings if None provided
        if df_settings is None:
            df_settings = list(DEFAULT_DF_SETTINGS.values())

        super().__init__(
            df_settings,
            min_sample_leaf,
            min_sample_alpha,
            min_sample_fit,
            max_depth,
            max_model_depth,
            max_features,
            0 if max_pivot is None else max_pivot,
            rel_tolerance,
            precision_scale,
        )
        self.feature_idx = feature_idx

    def predict(self, X):
        return super().predict(X[:, self.feature_idx])

    def tree_summary(self, feature_names: list | None = None) -> pd.DataFrame:
        df = pd.DataFrame(
            self.print(),
            columns=[
                "depth",
                "model_depth",
                "node_id",
                "node_type",
                "feature_index",
                "split_value",
                "intercept_left",
                "slope_left",
                "intercept_right",
                "slope_right",
                "rss_reduction",
            ],
        )

        df["node_type"] = df["node_type"].map(
            {0: "con", 1: "lin", 2: "pcon", 3: "blin", 4: "plin", 5: "pconc"}
        )

        if feature_names is not None:
            feature_names = np.array(feature_names)[self.feature_idx]
            df["feature_name"] = df["feature_index"].map(dict(enumerate(feature_names)))

        return df

    @property
    def feature_importances_(self):
        """Get feature importances for this tree.

        Returns normalized RSS reduction per feature (summing to 1.0).
        Follows sklearn's approach of normalizing per-tree importances.
        """
        raw_importance = super().feature_importances()
        total = raw_importance.sum()
        if total > 0:
            return raw_importance / total
        return raw_importance


class RaFFLE(BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 12,
        max_model_depth: int = 100,
        min_sample_fit: int = 10,
        min_sample_alpha: int = 5,
        min_sample_leaf: int = 5,
        random_state: int = 42,
        n_features_tree: float | str = 1.0,
        n_features_node: float | str = 1.0,
        df_settings: dict[str, int] | None = None,
        rel_tolerance: float = 0.01,
        precision_scale: float = 1e-10,
        alpha: float = 1,
        max_pivot: int | None = None,
    ):
        """
        Random Forest with PILOT trees as estimators.
        Args:
        - n_estimators (int): number of PILOT trees
        - max_depth (int): max depth to grow each PILOT tree (excl linear nodes)
        - max_model_depth (int): max depth to grow each PILOT tree (incl linear nodes)
        - min_sample_fit (int): min samples needed to fit any node
        - min_sample_alpha (int): min samples needed to fit a piecewise node
        - min_sample_leaf (int): min samples needed in each leaf node
        - random_state (int): seed used for bootstrapping and feature sampling
        - n_features_tree (float): relative share of features to consider on tree level
        - n_features_node (float): relative share of features to consider on node level
        - df_settings (Optional[dict]): optionally override the default settings.
            If not None, alpha is ignored
        - rel_tolerance (float): relative improvement in RSS needed to continue growing
        - precision_scale (float): precision scale
        - alpha (float): number between 0 and 1, sets the df to 1 + alpha * [0, 1, 4, 4, 6, 4].
            Ignored if df_settings is not None
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.min_sample_fit = min_sample_fit
        self.min_sample_alpha = min_sample_alpha
        self.min_sample_leaf = min_sample_leaf
        self.random_state = random_state
        self.n_features_tree = n_features_tree
        self.n_features_node = n_features_node
        self.df_settings = (
            list(df_settings.values())
            if df_settings is not None
            else (1 + alpha * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1)).tolist()
        )
        self.rel_tolerance = rel_tolerance
        self.precision_scale = precision_scale
        self.alpha = alpha
        self.max_pivot = max_pivot

    def fit(self, X, y, categorical_idx=None, n_workers: int = 1):
        """Fit a random forest ensemble of PILOT trees.

        Args:
            X (pd.DataFrame | np.ndarray): Feature data.
                Categorical features need to be label encoded.
            y (pd.Series | np.ndarray): Target values
            categorical_idx (Iterable[int] | None, optional):
                (numerical) indices of categorical features.
                If any index is -1, all features are considered numerical.
                Defaults to None, i.e. no all features are considered numerical.
            n_workers (int, optional): > 1 not supported a.t.m.. Defaults to 1.
        """

        categorical = np.zeros(X.shape[1], dtype=int)
        if categorical_idx is not None and not (categorical_idx == -1).any():
            categorical[categorical_idx] = 1

        X = np.array(X)
        y = np.array(y).flatten()
        n_features_tree = (
            int(np.sqrt(X.shape[1]))
            if self.n_features_tree == "sqrt"
            else int(X.shape[1] * self.n_features_tree)
        )
        n_features_node = min(
            n_features_tree,  # n_feature_node cannot be larger than n_features_tree
            (
                int(np.sqrt(X.shape[1]))
                if self.n_features_node == "sqrt"
                else int(X.shape[1] * self.n_features_node)
            ),
        )
        np.random.seed(self.random_state)
        self.estimators = [
            PILOTWrapper(
                feature_idx=np.random.choice(
                    np.arange(X.shape[1]), size=n_features_tree, replace=False
                ),
                df_settings=self.df_settings,
                min_sample_leaf=self.min_sample_leaf,
                min_sample_alpha=self.min_sample_alpha,
                min_sample_fit=self.min_sample_fit,
                max_depth=self.max_depth,
                max_model_depth=self.max_model_depth,
                max_features=n_features_node,
                max_pivot=self.max_pivot,
                rel_tolerance=self.rel_tolerance,
                precision_scale=self.precision_scale,
            )
            for _ in range(self.n_estimators)
        ]

        if n_workers == -1:
            n_workers = mp.cpu_count()
        if n_workers == 1:
            # avoid overhead of parallel processing
            self.estimators = [
                _fit_single_estimator(estimator, X, y, categorical) for estimator in self.estimators
            ]
        else:
            raise NotImplementedError("Parallel processing not available for CPILOT")
            with mp.Pool(processes=n_workers) as p:
                self.estimators = p.map(
                    partial(
                        _fit_single_estimator,
                        X=X,
                        y=y,
                        categorical_idx=categorical_idx,
                        n_features=n_features,
                    ),
                    self.estimators,
                )
        # filter failed estimators
        self.estimators = [e for e in self.estimators if e is not None]

    def predict(self, X, individual: bool = False) -> np.ndarray:
        X = np.array(X)
        predictions = np.concatenate([e.predict(X).reshape(-1, 1) for e in self.estimators], axis=1)
        if individual:
            return predictions
        return predictions.mean(axis=1)

    @property
    def feature_importances_(self):
        """Compute average feature importance across all trees in the forest.

        Feature importance is calculated as the normalized RSS reduction per tree,
        averaged across all trees, then re-normalized. This follows sklearn's approach:
        1. Each tree's importances are normalized (sum to 1.0)
        2. Importances are averaged across trees
        3. Final result is re-normalized (sum to 1.0)

        Features not selected in a particular tree contribute 0 to that tree's importance.

        Returns:
            np.ndarray: Array of shape (n_features,) with normalized feature importances
                       summing to 1.0.
        """
        if not hasattr(self, 'estimators') or len(self.estimators) == 0:
            raise ValueError("Model must be fitted before accessing feature_importances_")

        # Get the number of features from the first estimator
        n_features_total = len(self.estimators[0].feature_idx) if hasattr(self.estimators[0], 'feature_idx') else 0

        if n_features_total == 0:
            # Fallback: check from fitted data
            raise ValueError("Cannot determine number of features")

        # We need to know the total number of features in the original space
        # This is the max feature index across all trees + 1
        max_feature_idx = max(max(e.feature_idx) for e in self.estimators)
        n_features = max_feature_idx + 1

        # Initialize total importance array
        total_importance = np.zeros(n_features)

        # Aggregate normalized importance from each tree
        # Note: estimator.feature_importances_ already returns normalized values (sum to 1.0)
        for estimator in self.estimators:
            tree_importance = estimator.feature_importances_
            # Map the tree's feature importances back to the full feature space
            for local_idx, global_idx in enumerate(estimator.feature_idx):
                if local_idx < len(tree_importance):
                    total_importance[global_idx] += tree_importance[local_idx]

        # Average across trees
        avg_importance = total_importance / len(self.estimators)

        # Re-normalize to sum to 1.0 (sklearn's approach)
        total = avg_importance.sum()
        if total > 0:
            return avg_importance / total
        return avg_importance


def _fit_single_estimator(estimator, X: np.ndarray, y: np.ndarray, categorical_idx: np.ndarray):
    bootstrap_idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
    feature_idx = estimator.feature_idx
    categorical_idx = categorical_idx[feature_idx].astype(int)
    X_bootstrap = X[np.ix_(bootstrap_idx, feature_idx)]

    try:
        estimator.train(X_bootstrap, y[bootstrap_idx], categorical_idx)
        return estimator
    except ValueError as e:
        print(e)
        return None


# Backward compatibility aliases
RandomForestCPilot = RaFFLE
CPILOTWrapper = PILOTWrapper
