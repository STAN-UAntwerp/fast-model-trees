"""Tests for the RaFFLE random forest API."""
import numpy as np
import pytest
from pilot import RaFFLE, DEFAULT_DF_SETTINGS


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8))
    y = X[:, 0] + 0.3 * X[:, 2] ** 2 + rng.standard_normal(300) * 0.1
    return X, y


class TestRaFFLEBasic:
    def test_fit_predict(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=5, max_depth=4, random_state=1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))

    def test_default_construction(self, simple_data):
        """RaFFLE() with defaults should work end-to-end."""
        X, y = simple_data
        model = RaFFLE()
        model.fit(X, y)
        assert model.predict(X).shape == (len(y),)

    def test_individual_predictions(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=4, random_state=7)
        model.fit(X, y)
        ind = model.predict(X, individual=True)
        assert ind.shape == (len(y), 4)

    def test_reproducible_with_seed(self, simple_data):
        X, y = simple_data
        m1 = RaFFLE(n_estimators=5, random_state=99)
        m1.fit(X, y)
        m2 = RaFFLE(n_estimators=5, random_state=99)
        m2.fit(X, y)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))

    def test_custom_df_settings(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=3, df_settings=DEFAULT_DF_SETTINGS, random_state=0)
        model.fit(X, y)
        assert model.predict(X).shape == (len(y),)

    def test_feature_importances_shape(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=5, random_state=2)
        model.fit(X, y)
        imp = model.feature_importances_
        assert imp.shape == (X.shape[1],)
        assert pytest.approx(imp.sum(), abs=1e-6) == 1.0
        assert np.all(imp >= 0)

    def test_sqrt_features(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=3, n_features_tree="sqrt", n_features_node="sqrt", random_state=5)
        model.fit(X, y)
        assert model.predict(X).shape == (len(y),)
