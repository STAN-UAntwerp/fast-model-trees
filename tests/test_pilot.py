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
