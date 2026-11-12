"""Tests for the RaFFLE random forest API."""
import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, r2_score
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

    def test_sklearn_compatible(self, simple_data):
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score

        X, y = simple_data
        model = RaFFLE(n_estimators=3, alpha=0.5, random_state=0)
        assert clone(model).get_params() == model.get_params()
        scores = cross_val_score(model, X, y, cv=3)
        assert scores.shape == (3,)
        assert model.fit(X, y).score(X, y) > 0.5

    def test_df_settings_as_list(self, simple_data):
        X, y = simple_data
        settings = list(DEFAULT_DF_SETTINGS.values())
        model = RaFFLE(n_estimators=2, df_settings=settings, random_state=0).fit(X, y)
        assert model.df_settings_ == settings
        assert model.predict(X).shape == (len(y),)

    def test_alpha_scales_df_settings(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=2, alpha=0.5, random_state=0).fit(X, y)
        assert model.df_settings_ == [1.0, 1.5, 3.0, 3.0, 4.0, 3.0]
        dict_model = RaFFLE(n_estimators=2, df_settings=DEFAULT_DF_SETTINGS, alpha=0.5).fit(X, y)
        assert dict_model.df_settings_ == list(DEFAULT_DF_SETTINGS.values())

    def test_feature_importances_shape(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=5, random_state=2)
        model.fit(X, y)
        imp = model.feature_importances_
        assert imp.shape == (X.shape[1],)
        assert pytest.approx(imp.sum(), abs=1e-6) == 1.0
        assert np.all(imp >= 0)

    def test_feature_importances_shape_with_feature_subsets(self, simple_data):
        """All features get an importance, also those that no tree was trained on."""
        X, y = simple_data
        X = np.column_stack([X, np.zeros((len(X), 20))])
        model = RaFFLE(n_estimators=2, n_features_tree=0.1, random_state=0).fit(X, y)
        used = np.unique(np.concatenate([e.feature_idx for e in model.estimators]))
        assert used.max() < X.shape[1] - 1  # the last column is not used by any tree
        assert model.feature_importances_.shape == (X.shape[1],)

    def test_sqrt_features(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=3, n_features_tree="sqrt", n_features_node="sqrt", random_state=5)
        model.fit(X, y)
        assert model.predict(X).shape == (len(y),)

    def test_categorical_idx_as_list(self):
        rng = np.random.default_rng(3)
        X = np.column_stack([rng.standard_normal(200), rng.integers(0, 4, 200).astype(float)])
        y = X[:, 0] + X[:, 1] + rng.standard_normal(200) * 0.1
        model = RaFFLE(n_estimators=3, random_state=0).fit(X, y, categorical_idx=[1])
        assert model.predict(X).shape == (len(y),)


class TestRaFFLEBootstrap:
    def test_bootstrap_idx_stored(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=5, random_state=1).fit(X, y)
        assert model.n_samples_fit_ == len(X)
        samples = model.estimators_samples_
        assert len(samples) == len(model.estimators)
        for estimator, idx in zip(model.estimators, samples):
            assert idx is estimator.bootstrap_idx
            assert idx.shape == (len(X),)
            assert idx.min() >= 0 and idx.max() < len(X)
        # bootstrap samples differ between trees
        assert not np.array_equal(samples[0], samples[1])

    def test_bootstrap_idx_reproducible(self, simple_data):
        X, y = simple_data
        s1 = RaFFLE(n_estimators=4, random_state=11).fit(X, y).estimators_samples_
        s2 = RaFFLE(n_estimators=4, random_state=11).fit(X, y).estimators_samples_
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    def test_bootstrap_idx_matches_tree(self, simple_data):
        """Each tree is trained on exactly its stored bootstrap sample."""
        X, y = simple_data
        model = RaFFLE(n_estimators=3, random_state=4).fit(X, y)
        for estimator in model.estimators:
            root = estimator.tree_summary().iloc[0]
            assert root["n_samples"] == len(estimator.bootstrap_idx)

    def test_oob_mask(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=5, random_state=1).fit(X, y)
        mask = model.oob_mask()
        assert mask.shape == (len(X), 5)
        for j, idx in enumerate(model.estimators_samples_):
            expected = np.ones(len(X), dtype=bool)
            expected[idx] = False
            np.testing.assert_array_equal(mask[:, j], expected)
        # roughly (1 - 1/n)^n ~ 36.8% of rows are out-of-bag for each tree
        assert 0.25 < mask.mean() < 0.5

    def test_oob_predict_matches_manual(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=8, random_state=3).fit(X, y)
        mask = model.oob_mask()
        individual = model.predict(X, individual=True)
        n_trees = mask.sum(axis=1)
        with np.errstate(invalid="ignore"):
            expected = np.where(n_trees > 0, (individual * mask).sum(axis=1) / n_trees, np.nan)
        np.testing.assert_allclose(model.oob_predict(X), expected, equal_nan=True)

    def test_oob_predict_rejects_other_data(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=3, random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="training data"):
            model.oob_predict(X[:10])

    def test_oob_score(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=20, max_depth=4, random_state=0, oob_score=True).fit(X, y)
        oob = model.oob_prediction_
        assert oob.shape == (len(y),)
        assert not np.isnan(oob).any()
        np.testing.assert_allclose(oob, model.oob_predict(X))
        assert model.oob_score_ == pytest.approx(r2_score(y, oob))
        # OOB is an honest estimate: worse than the in-sample fit, but still informative
        assert 0.5 < model.oob_score_ < r2_score(y, model.predict(X))

    def test_oob_score_callable(self, simple_data):
        X, y = simple_data
        model = RaFFLE(
            n_estimators=20, max_depth=4, random_state=0, oob_score=mean_squared_error
        ).fit(X, y)
        assert model.oob_score_ == pytest.approx(mean_squared_error(y, model.oob_prediction_))

    def test_oob_score_warns_for_rows_without_oob(self, simple_data):
        X, y = simple_data
        with pytest.warns(UserWarning, match="no OOB prediction"):
            model = RaFFLE(n_estimators=2, random_state=0, oob_score=True).fit(X, y)
        has_oob = ~np.isnan(model.oob_prediction_)
        assert model.oob_score_ == pytest.approx(r2_score(y[has_oob], model.oob_prediction_[has_oob]))

    def test_no_oob_attributes_by_default(self, simple_data):
        X, y = simple_data
        model = RaFFLE(n_estimators=3, random_state=0).fit(X, y)
        assert not hasattr(model, "oob_score_")
        assert not hasattr(model, "oob_prediction_")
