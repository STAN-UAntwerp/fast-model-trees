"""Tests for the PILOT single-tree API."""
import numpy as np
import pytest
from pilot import PILOT, DEFAULT_DF_SETTINGS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.standard_normal(200) * 0.1
    return X, y


@pytest.fixture
def categorical_data():
    rng = np.random.default_rng(0)
    n = 600
    X = np.column_stack(
        [rng.standard_normal(n), rng.integers(0, 6, n).astype(float), rng.standard_normal(n)]
    )
    level_effect = np.array([3.0, -2.0, 0.5, 3.0, -2.0, 1.0])
    y = (
        level_effect[X[:, 1].astype(int)]
        + X[:, 0]
        + np.where(X[:, 2] > 0, 2 * X[:, 2], 0)
        + rng.standard_normal(n) * 0.1
    )
    return X, y, [0, 1, 0]


# ---------------------------------------------------------------------------
# Constructor / API
# ---------------------------------------------------------------------------

class TestPILOTConstructor:
    def test_default_construction(self):
        """PILOT() with no arguments should not raise."""
        model = PILOT()
        assert model.df_settings == list(DEFAULT_DF_SETTINGS.values())
        assert model.max_features is None

    def test_keyword_args(self, simple_data):
        """The reported bug: keyword args must work."""
        X, y = simple_data
        model = PILOT(
            df_settings=list(DEFAULT_DF_SETTINGS.values()),
            max_depth=5,
            max_features=X.shape[1],
        )
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_default_df_settings_no_import(self, simple_data):
        """Users should not need to pass df_settings at all."""
        X, y = simple_data
        model = PILOT(max_depth=5)
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_max_features_none_uses_all(self, simple_data):
        """max_features=None should default to all features at train time."""
        X, y = simple_data
        model = PILOT(max_features=None)
        model.train(X, y)
        assert model.predict(X).shape == (len(y),)

    def test_max_features_explicit(self, simple_data):
        X, y = simple_data
        model = PILOT(max_features=3)
        model.train(X, y)
        assert model.predict(X).shape == (len(y),)

    def test_disable_node_type_with_minus_one(self, simple_data):
        """Setting a df_settings entry to -1 disables that node type."""
        X, y = simple_data
        dfs = list(DEFAULT_DF_SETTINGS.values())
        dfs[3] = -1  # disable blin
        model = PILOT(df_settings=dfs)
        model.train(X, y)
        assert model.predict(X).shape == (len(y),)

    def test_categorical_features(self, simple_data):
        X, y = simple_data
        categorical = np.zeros(X.shape[1], dtype=int)
        model = PILOT(max_depth=3)
        model.train(X, y, categorical=categorical)
        assert model.predict(X).shape == (len(y),)


# ---------------------------------------------------------------------------
# train / predict behaviour
# ---------------------------------------------------------------------------

class TestTrainPredict:
    def test_predict_before_train_raises(self):
        model = PILOT()
        with pytest.raises(ValueError, match="train"):
            model.predict(np.zeros((5, 3)))

    def test_train_returns_self(self, simple_data):
        X, y = simple_data
        model = PILOT()
        result = model.train(X, y)
        assert result is model

    def test_predictions_are_finite(self, simple_data):
        X, y = simple_data
        model = PILOT(max_depth=5)
        model.train(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

    def test_retrain_replaces_tree(self, simple_data):
        X, y = simple_data
        model = PILOT(max_depth=3)
        model.train(X, y)
        preds_first = model.predict(X).copy()
        model.train(X, y * 2)
        preds_second = model.predict(X)
        # predictions should differ after retraining on scaled target
        assert not np.allclose(preds_first, preds_second)

    def test_pandas_input(self, simple_data):
        """PILOT should handle pandas DataFrames/Series."""
        pd = pytest.importorskip("pandas")
        X, y = simple_data
        X_df = pd.DataFrame(X)
        y_s = pd.Series(y)
        model = PILOT(max_depth=3)
        model.train(X_df, y_s)
        assert model.predict(X_df).shape == (len(y),)


# ---------------------------------------------------------------------------
# tree_summary
# ---------------------------------------------------------------------------

class TestTreeSummary:
    def test_tree_summary_before_train_raises(self):
        model = PILOT()
        with pytest.raises(ValueError, match="train"):
            model.tree_summary()

    def test_tree_summary_returns_dataframe(self, simple_data):
        pd = pytest.importorskip("pandas")
        X, y = simple_data
        model = PILOT(max_depth=3)
        model.train(X, y)
        df = model.tree_summary()
        assert isinstance(df, pd.DataFrame)
        assert "node_type" in df.columns

    def test_tree_summary_with_feature_names(self, simple_data):
        pytest.importorskip("pandas")
        X, y = simple_data
        names = [f"feat_{i}" for i in range(X.shape[1])]
        model = PILOT(max_depth=3)
        model.train(X, y)
        df = model.tree_summary(feature_names=names)
        assert "feature_name" in df.columns

    def test_tree_summary_pconc_pivot_values(self, categorical_data):
        X, y, categorical = categorical_data
        model = PILOT(max_depth=4)
        model.train(X, y, categorical=categorical)
        df = model.tree_summary()
        is_pconc = df["node_type"] == "pconc"
        assert is_pconc.any()
        levels = np.unique(X[:, 1])
        for pivots in df.loc[is_pconc, "pivot_values"]:
            assert 0 < len(pivots) < len(levels)
            assert np.isin(pivots, levels).all()
            assert (np.diff(pivots) > 0).all()
        assert df.loc[~is_pconc, "pivot_values"].isna().all()
        assert df.loc[is_pconc, ["split_value", "slope_left", "slope_right"]].isna().all().all()

    def test_tree_summary_parent_node_id(self, categorical_data):
        X, y, categorical = categorical_data
        model = PILOT(max_depth=4)
        model.train(X, y, categorical=categorical)
        df = model.tree_summary()
        assert df["node_id"].is_unique
        assert df["parent_node_id"].isna().sum() == 1
        assert df["parent_node_id"].iloc[0] is None
        assert df["parent_node_id"].iloc[1:].isin(df["node_id"]).all()

        nodes = df.set_index("node_id")
        n_children = {"con": 0, "lin": 1, "pcon": 2, "blin": 2, "plin": 2, "pconc": 2}
        children = df.iloc[1:].groupby("parent_node_id").size()
        for node_id, row in nodes.iterrows():
            assert children.get(node_id, 0) == n_children[row["node_type"]]
        parent_depth = df["parent_node_id"].iloc[1:].map(nodes["model_depth"])
        assert (parent_depth.to_numpy() == df["model_depth"].iloc[1:].to_numpy() - 1).all()

    def test_tree_summary_reconstructs_predictions(self, categorical_data):
        """The summary alone should be enough to rebuild the tree and predict with it."""
        X, y, categorical = categorical_data
        model = PILOT(max_depth=4)
        model.train(X, y, categorical=categorical)
        df = model.tree_summary()

        children = {}
        for _, row in df.iloc[1:].iterrows():  # left child is listed before right child
            children.setdefault(row["parent_node_id"], []).append(row)

        def predict_one(x):
            node, pred = df.iloc[0], 0.0
            while node["node_type"] != "con":
                kids = children[node["node_id"]]
                xj = x[int(node["feature_index"])]
                if node["node_type"] == "lin":
                    go_left = True
                elif node["node_type"] == "pconc":
                    go_left = xj in node["pivot_values"]
                else:
                    go_left = xj <= node["split_value"]
                side = "left" if go_left else "right"
                slope = node[f"slope_{side}"]
                pred += node[f"intercept_{side}"] + (0.0 if np.isnan(slope) else slope * xj)
                pred = np.clip(pred, y.min(), y.max())
                node = kids[0 if go_left else 1]
            return np.clip(pred + node["intercept_left"], y.min(), y.max())

        expected = model.predict(X)
        actual = np.array([predict_one(x) for x in X])
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# feature_importances_
# ---------------------------------------------------------------------------

class TestFeatureImportances:
    def test_feature_importances_before_train_raises(self):
        model = PILOT()
        with pytest.raises(ValueError, match="train"):
            _ = model.feature_importances_

    def test_feature_importances_sum_to_one(self, simple_data):
        X, y = simple_data
        model = PILOT(max_depth=5)
        model.train(X, y)
        imp = model.feature_importances_
        assert imp.shape == (X.shape[1],)
        assert pytest.approx(imp.sum(), abs=1e-6) == 1.0

    def test_feature_importances_non_negative(self, simple_data):
        X, y = simple_data
        model = PILOT(max_depth=5)
        model.train(X, y)
        assert np.all(model.feature_importances_ >= 0)


# ---------------------------------------------------------------------------
# DEFAULT_DF_SETTINGS
# ---------------------------------------------------------------------------

class TestDefaultDfSettings:
    def test_keys(self):
        assert set(DEFAULT_DF_SETTINGS.keys()) == {"con", "lin", "pcon", "blin", "plin", "pconc"}

    def test_values_positive(self):
        assert all(v > 0 for v in DEFAULT_DF_SETTINGS.values())
