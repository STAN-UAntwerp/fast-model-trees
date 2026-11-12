from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from .cpilot import PILOT as _CRawPILOT
from .constants import DEFAULT_DF_SETTINGS


class PILOT:
    """Single PILOT tree with a user-friendly Python API.

    Parameters
    ----------
    df_settings : list of float, optional
        Degrees of freedom for each node type: [con, lin, pcon, blin, plin, pconc].
        Defaults to ``list(DEFAULT_DF_SETTINGS.values())`` = [1, 2, 5, 5, 7, 5].
        Set a value to -1 to disable that node type entirely.
    max_depth : int, optional
        Maximum split depth (excluding linear model nodes). Default: 20.
    max_model_depth : int, optional
        Maximum total depth including linear model nodes. Default: 100.
    max_features : int or None, optional
        Number of features to consider at each split.
        ``None`` (default) means use all features, resolved at ``train()`` time.
    min_sample_leaf : int, optional
        Minimum samples required in each leaf node. Default: 5.
    min_sample_alpha : int, optional
        Minimum samples required to fit a piecewise node. Default: 5.
    min_sample_fit : int, optional
        Minimum samples required to fit any node. Default: 5.
    max_pivot : int or None, optional
        Maximum pivots per feature for approximate splits. ``None`` (default)
        disables approximation. Note: approximation cannot be used with the
        blin node type (i.e. when ``df_settings[3] >= 0``).
    rel_tolerance : float, optional
        Minimum relative RSS improvement required to continue growing. Default: 0.01.
    precision_scale : float, optional
        Numerical precision scale. Default: 1e-10.
    """

    def __init__(
        self,
        df_settings=None,
        max_depth=20,
        max_model_depth=100,
        max_features=None,
        min_sample_leaf=5,
        min_sample_alpha=5,
        min_sample_fit=5,
        max_pivot=None,
        rel_tolerance=0.01,
        precision_scale=1e-10,
    ):
        if df_settings is None:
            df_settings = list(DEFAULT_DF_SETTINGS.values())
        self.df_settings = list(df_settings)
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.max_features = max_features
        self.min_sample_leaf = min_sample_leaf
        self.min_sample_alpha = min_sample_alpha
        self.min_sample_fit = min_sample_fit
        self.max_pivot = max_pivot
        self.rel_tolerance = rel_tolerance
        self.precision_scale = precision_scale
        self._tree = None

    def train(self, X, y, categorical=None):
        """Fit the PILOT tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Categorical features must be label-encoded.
        y : array-like of shape (n_samples,)
            Target values.
        categorical : array-like of int of shape (n_features,), optional
            Binary indicator per feature: 1 = categorical, 0 = numerical.
            Defaults to all-numerical.

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).flatten()
        n_features = X.shape[1]
        max_features = self.max_features if self.max_features is not None else n_features

        if categorical is None:
            categorical = np.zeros(n_features, dtype=np.uint32)
        else:
            categorical = np.array(categorical, dtype=np.uint32)

        self._tree = _CRawPILOT(
            self.df_settings,
            self.min_sample_leaf,
            self.min_sample_alpha,
            self.min_sample_fit,
            self.max_depth,
            self.max_model_depth,
            max_features,
            0 if self.max_pivot is None else self.max_pivot,
            self.rel_tolerance,
            self.precision_scale,
        )
        self._tree.train(X, y, categorical)
        return self

    def predict(self, X):
        """Return predictions for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self._tree is None:
            raise ValueError("Model must be trained before predicting. Call train() first.")
        return self._tree.predict(np.array(X, dtype=float))

    def tree_summary(self, feature_names=None):
        """Return a DataFrame describing each node in the fitted tree.

        The tree is listed in pre-order: a node is followed by its left subtree and
        then its right subtree. Nodes are identified by ``node_id``; together with
        ``parent_node_id`` this is enough to reconstruct the tree.

        Parameters
        ----------
        feature_names : list of str, optional
            Column names of the training data. When provided, a ``feature_name``
            column is added to the returned DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per node, with columns:

            depth
                Number of splits (pcon/blin/plin/pconc) above the node. lin nodes
                do not increase the depth. This is what ``max_depth`` limits.
            model_depth
                Number of nodes above the node, lin nodes included (root = 0).
                This is what ``max_model_depth`` limits.
            node_id
                Identifier of the node, unique within the tree. It is the string
                ``"<model_depth>_<k>"``, with ``k`` the index of the node among
                the nodes with the same ``model_depth`` (e.g. ``"2_3"``); the
                root is ``"0_0"``.
            parent_node_id
                ``node_id`` of the parent. None for the root. A lin node has a
                single child; a split node has two, of which the left child is
                listed first (and has the smaller index ``k``).
            node_type
                Model fitted in the node: ``con`` (constant, always a leaf),
                ``lin`` (linear model on all observations, no split), ``pcon``
                (piecewise constant), ``blin`` (broken linear, continuous at the
                split), ``plin`` (piecewise linear) or ``pconc`` (piecewise
                constant on a categorical feature).
            feature_index
                Index of the feature used by the node's model. NaN for con nodes.
            split_value
                Split point for pcon/blin/plin: observations with
                ``x <= split_value`` go to the left child, the others to the right.
                NaN for con, lin and pconc nodes.
            pivot_values
                For pconc nodes, sorted array of the levels of the categorical
                feature that go to the left child; all other levels go to the
                right. None for all other node types.
            intercept_left, slope_left
                Coefficients of the model for the observations going left, which
                contribute ``intercept_left + slope_left * x`` to the prediction.
                For lin nodes this is the model for all observations; for con
                nodes ``intercept_left`` is the constant. The slope is 0 for
                pcon and NaN for con and pconc (no slope term).
            intercept_right, slope_right
                Same, for the observations going right. NaN for con and lin nodes.
            rss_reduction
                Reduction in residual sum of squares obtained by the node's model
                (the basis of ``feature_importances_``).
            n_samples
                Number of training observations that reached the node. For a con
                leaf this is the number of samples in the leaf. A lin node and its
                child have the same count; for a split node the counts of the two
                children add up to the count of the node. In RaFFLE, trees are
                trained on a bootstrap sample, so an observation drawn several
                times is counted several times.
            feature_name
                Name of ``feature_index``; only present if ``feature_names`` is given.

        Notes
        -----
        A prediction is the sum of the contributions of all nodes on the path from
        the root to a con leaf. During prediction, ``x`` is first clipped to the
        range of the feature seen in that node during training and the running sum
        is clipped to the range of the training response; these ranges are not part
        of the summary.
        """
        if self._tree is None:
            raise ValueError("Model must be trained before calling tree_summary(). Call train() first.")
        df = _tree_summary_frame(self._tree)
        if feature_names is not None:
            feature_names = np.array(feature_names)
            df["feature_name"] = df["feature_index"].map(dict(enumerate(feature_names)))
        return df

    @property
    def feature_importances_(self):
        """Normalized RSS reduction per feature (sums to 1.0)."""
        if self._tree is None:
            raise ValueError(
                "Model must be trained before accessing feature_importances_. Call train() first."
            )
        raw = self._tree.feature_importances()
        total = raw.sum()
        return raw / total if total > 0 else raw


class PILOTWrapper(_CRawPILOT):
    """A PILOT tree trained on a subset of the features: the trees of a RaFFLE forest.

    Takes the same parameters as :class:`PILOT`, plus ``feature_idx``, and
    ``max_features`` is required. Unlike :class:`PILOT`, :meth:`train` must be given
    only the columns in ``feature_idx`` (in that order), while :meth:`predict` takes
    the full feature matrix and selects those columns itself.

    Attributes
    ----------
    feature_idx : np.ndarray
        Columns of the full feature matrix the tree uses.
    bootstrap_idx : np.ndarray
        Only for trees of a fitted :class:`RaFFLE`: rows of the training data in the
        bootstrap sample the tree was trained on.
    """

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
        """Return predictions for the full feature matrix X."""
        return super().predict(X[:, self.feature_idx])

    def tree_summary(self, feature_names: list | None = None) -> pd.DataFrame:
        """Return a DataFrame describing each node in the fitted tree.

        See :meth:`PILOT.tree_summary` for the meaning of the columns. Note that
        ``feature_index`` refers to the position within ``self.feature_idx`` (the
        features sampled for this tree); ``feature_name`` is mapped back to the
        original features.
        """
        df = _tree_summary_frame(self)


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


class RaFFLE(RegressorMixin, BaseEstimator):
    """Random Forest Featuring Linear Extensions: a random forest of PILOT trees.

    Every tree is a PILOT tree trained on a bootstrap sample of the training data
    and a random subset of the features. Predictions are the average of the
    predictions of the trees.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees in the forest. Default: 10.
    max_depth : int, optional
        Maximum split depth of each tree (excluding linear model nodes). Default: 12.
    max_model_depth : int, optional
        Maximum total depth of each tree, linear model nodes included. Default: 100.
    min_sample_fit : int, optional
        Minimum samples required to fit any node. Default: 10.
    min_sample_alpha : int, optional
        Minimum samples required to fit a piecewise node. Default: 5.
    min_sample_leaf : int, optional
        Minimum samples required in each leaf node. Default: 5.
    random_state : int, optional
        Seed used for the bootstrap samples and the feature sampling. Default: 42.
    n_features_tree : float or "sqrt", optional
        Share of the features each tree is trained on, or ``"sqrt"`` for
        ``int(sqrt(n_features))`` features. Default: 1.0.
    n_features_node : float or "sqrt", optional
        Share of the features considered at each node (capped at the number of
        features of the tree), or ``"sqrt"``. Default: 1.0.
    df_settings : dict of str to float or list of float, optional
        Degrees of freedom per node type: a dict with the keys of
        :data:`pilot.DEFAULT_DF_SETTINGS` in the same order, or a list
        ``[con, lin, pcon, blin, plin, pconc]``. Overrides ``alpha``.
    rel_tolerance : float, optional
        Minimum relative RSS improvement required to continue growing. Default: 0.01.
    precision_scale : float, optional
        Numerical precision scale. Default: 1e-10.
    alpha : float, optional
        Number between 0 and 1 that scales the complexity penalty of the node
        types: the degrees of freedom are ``1 + alpha * (DEFAULT_DF_SETTINGS - 1)``,
        i.e. ``1 + alpha * [0, 1, 4, 4, 6, 4]``. Ignored if ``df_settings`` is given.
        Default: 1.
    max_pivot : int or None, optional
        Maximum pivots per feature for approximate splits. ``None`` (default)
        disables approximation.
    oob_score : bool or callable, optional
        Whether to compute out-of-bag predictions and an out-of-bag score at the
        end of :meth:`fit`. ``True`` uses the R² score; a callable
        ``score(y_true, y_pred)`` (e.g. ``sklearn.metrics.mean_squared_error``)
        is used as the metric instead. Default: False.

    Attributes
    ----------
    estimators : list of PILOTWrapper
        The fitted trees. Each tree has a ``feature_idx`` attribute (the features
        it was trained on) and a ``bootstrap_idx`` attribute (the rows of the
        training data in its bootstrap sample).
    df_settings_ : list of float
        Degrees of freedom of ``[con, lin, pcon, blin, plin, pconc]`` used by the
        trees, resolved from ``df_settings`` or ``alpha``.
    n_samples_fit_ : int
        Number of rows in the training data.
    n_features_in_ : int
        Number of features in the training data.
    oob_prediction_ : np.ndarray of shape (n_samples,)
        Out-of-bag prediction for every training row: the average prediction of
        the trees whose bootstrap sample does not contain that row. NaN for rows
        that are in the bootstrap sample of every tree. Only set when
        ``oob_score`` is enabled.
    oob_score_ : float
        Score of ``oob_prediction_`` against the training targets, computed on
        the rows that have an out-of-bag prediction. Only set when ``oob_score``
        is enabled.
    """

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
        df_settings: dict[str, float] | list[float] | None = None,
        rel_tolerance: float = 0.01,
        precision_scale: float = 1e-10,
        alpha: float = 1,
        max_pivot: int | None = None,
        oob_score: bool | Callable = False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.min_sample_fit = min_sample_fit
        self.min_sample_alpha = min_sample_alpha
        self.min_sample_leaf = min_sample_leaf
        self.random_state = random_state
        self.n_features_tree = n_features_tree
        self.n_features_node = n_features_node
        self.df_settings = df_settings
        self.rel_tolerance = rel_tolerance
        self.precision_scale = precision_scale
        self.alpha = alpha
        self.max_pivot = max_pivot
        self.oob_score = oob_score

    def fit(self, X, y, categorical_idx=None, n_workers: int = 1):
        """Fit a random forest ensemble of PILOT trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature data. Categorical features need to be label encoded.
        y : array-like of shape (n_samples,)
            Target values.
        categorical_idx : array-like of int, optional
            Column indices of the categorical features. If any index is -1, all
            features are considered numerical. Defaults to None, i.e. all features
            are considered numerical.
        n_workers : int, optional
            Values other than 1 are not supported at the moment. Default: 1.

        Returns
        -------
        self
        """

        categorical = np.zeros(X.shape[1], dtype=int)
        if categorical_idx is not None:
            categorical_idx = np.asarray(categorical_idx, dtype=int)
            if not (categorical_idx == -1).any():
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
        if self.df_settings is None:
            df_settings = 1 + self.alpha * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1)
        elif isinstance(self.df_settings, dict):
            df_settings = self.df_settings.values()
        else:
            df_settings = self.df_settings
        self.df_settings_ = [float(v) for v in df_settings]

        np.random.seed(self.random_state)
        self.estimators = [
            PILOTWrapper(
                feature_idx=np.random.choice(
                    np.arange(X.shape[1]), size=n_features_tree, replace=False
                ),
                df_settings=self.df_settings_,
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
        # filter failed estimators
        self.estimators = [e for e in self.estimators if e is not None]
        self.n_samples_fit_ = len(X)
        self.n_features_in_ = X.shape[1]

        if self.oob_score:
            self.oob_prediction_ = self.oob_predict(X)
            has_oob = ~np.isnan(self.oob_prediction_)
            if not has_oob.any():
                raise ValueError(
                    "No training row is out-of-bag for any tree, the OOB score cannot be "
                    "computed. Increase n_estimators."
                )
            if not has_oob.all():
                warnings.warn(
                    f"{(~has_oob).sum()} training rows are in the bootstrap sample of every tree "
                    "and have no OOB prediction; they are left out of oob_score_. "
                    "Increase n_estimators to avoid this.",
                    UserWarning,
                )
            score = self.oob_score if callable(self.oob_score) else r2_score
            self.oob_score_ = score(y[has_oob], self.oob_prediction_[has_oob])
        return self

    def predict(self, X, individual: bool = False) -> np.ndarray:
        """Predict the target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        individual : bool, optional
            If True, return the prediction of every tree instead of their average.

        Returns
        -------
        np.ndarray of shape (n_samples,), or (n_samples, n_estimators) if ``individual``
        """
        X = np.array(X)
        predictions = np.concatenate([e.predict(X).reshape(-1, 1) for e in self.estimators], axis=1)
        if individual:
            return predictions
        return predictions.mean(axis=1)

    @property
    def estimators_samples_(self) -> list[np.ndarray]:
        """Bootstrap indices of every tree, in the order of ``estimators``.

        Each entry holds the row indices of the training data the tree was trained
        on, drawn with replacement (so it contains duplicates).
        """
        if not hasattr(self, "estimators"):
            raise ValueError("Model must be fitted before accessing estimators_samples_")
        return [e.bootstrap_idx for e in self.estimators]

    def oob_mask(self) -> np.ndarray:
        """Boolean matrix telling which training rows are out-of-bag for which tree.

        Returns
        -------
        np.ndarray of bool, shape (n_samples, n_estimators)
            Entry ``(i, j)`` is True if training row ``i`` is not in the bootstrap
            sample of tree ``j``.
        """
        if not hasattr(self, "estimators"):
            raise ValueError("Model must be fitted before calling oob_mask()")
        mask = np.ones((self.n_samples_fit_, len(self.estimators)), dtype=bool)
        for j, idx in enumerate(self.estimators_samples_):
            mask[idx, j] = False
        return mask

    def oob_predict(self, X) -> np.ndarray:
        """Out-of-bag predictions for the training data.

        Every row is predicted by averaging only the trees that did not see it
        during training. Combine with any metric to get an OOB estimate of the
        generalization error, e.g.
        ``mean_squared_error(y[mask], oob[mask])`` with ``mask = ~np.isnan(oob)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data passed to :meth:`fit`, with the rows in the same order.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            OOB prediction of each row; NaN for rows that are in the bootstrap
            sample of every tree.
        """
        X = np.array(X)
        if not hasattr(self, "estimators"):
            raise ValueError("Model must be fitted before calling oob_predict()")
        if len(X) != self.n_samples_fit_:
            raise ValueError(
                f"X has {len(X)} rows but the model was fitted on {self.n_samples_fit_} rows; "
                "oob_predict() needs the training data."
            )
        mask = self.oob_mask()
        predictions = np.zeros(len(X))
        n_trees = mask.sum(axis=1)
        for j, estimator in enumerate(self.estimators):
            rows = mask[:, j]
            if rows.any():
                predictions[rows] += estimator.predict(X[rows])
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_trees > 0, predictions / n_trees, np.nan)

    @property
    def feature_importances_(self):
        """Average feature importance across all trees in the forest.

        Follows scikit-learn's approach:

        1. each tree's RSS reductions per feature are normalized to sum to 1,
        2. the normalized importances are averaged across trees,
        3. the result is re-normalized to sum to 1.

        Features a tree was not trained on contribute 0 to that tree's importance.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Normalized feature importances, summing to 1.
        """
        if not hasattr(self, 'estimators') or len(self.estimators) == 0:
            raise ValueError("Model must be fitted before accessing feature_importances_")

        n_features = self.n_features_in_

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


def _tree_summary_frame(tree: _CRawPILOT) -> pd.DataFrame:
    """Build the tree summary of a fitted C++ tree (see ``PILOT.tree_summary``)."""
    df = pd.DataFrame(
        tree.print(),
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
            "parent_node_id",
            "n_samples",
        ],
    )
    df["node_type"] = df["node_type"].map(
        {0: "con", 1: "lin", 2: "pcon", 3: "blin", 4: "plin", 5: "pconc"}
    )
    is_pconc = df["node_type"] == "pconc"
    # the C++ node ids are only unique within a model depth: prefix them with the model depth
    model_depth = df["model_depth"].astype(int)
    is_root = df["parent_node_id"].isna()
    parent_node_id = (model_depth - 1).astype(str) + "_" + df["parent_node_id"].fillna(0).astype(int).astype(str)
    df["node_id"] = model_depth.astype(str) + "_" + df["node_id"].astype(int).astype(str)
    df["parent_node_id"] = parent_node_id.astype(object).where(~is_root, None)
    df["n_samples"] = df["n_samples"].astype(int)
    df["pivot_values"] = pd.Series(
        [p if pconc else None for p, pconc in zip(tree.pivots(), is_pconc)],
        index=df.index,
        dtype=object,
    )
    return df[
        [
            "depth",
            "model_depth",
            "node_id",
            "parent_node_id",
            "node_type",
            "feature_index",
            "split_value",
            "pivot_values",
            "intercept_left",
            "slope_left",
            "intercept_right",
            "slope_right",
            "rss_reduction",
            "n_samples",
        ]
    ]


def _fit_single_estimator(estimator, X: np.ndarray, y: np.ndarray, categorical_idx: np.ndarray):
    bootstrap_idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
    feature_idx = estimator.feature_idx
    categorical_idx = categorical_idx[feature_idx].astype(int)
    X_bootstrap = X[np.ix_(bootstrap_idx, feature_idx)]

    try:
        estimator.train(X_bootstrap, y[bootstrap_idx], categorical_idx)
        estimator.bootstrap_idx = bootstrap_idx
        return estimator
    except ValueError as e:
        print(e)
        return None


# Backward compatibility aliases
RandomForestCPilot = RaFFLE
CPILOTWrapper = PILOTWrapper
