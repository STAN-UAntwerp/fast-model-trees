"""
Test script for feature importance functionality in PILOT trees.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from pilot.c_ensemble import RandomForestCPilot, CPILOTWrapper

def test_single_tree_feature_importance():
    """Test feature importance for a single PILOT tree."""
    print("=" * 80)
    print("Testing single tree feature importance")
    print("=" * 80)

    # Create a simple dataset with informative features
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )

    # Train a single tree
    tree = CPILOTWrapper(
        feature_idx=np.arange(10),
        max_features=10,
        max_depth=5,
        min_sample_leaf=5
    )

    categorical_idx = np.zeros(10, dtype=int)
    tree.train(X, y, categorical_idx)

    # Get feature importance
    importance = tree.feature_importances_

    # Print tree summary
    print("\nTree Summary:")
    print(tree.tree_summary())

    print(f"Feature importances shape: {importance.shape}")
    print(f"Feature importances:\n{importance}")
    print(f"Sum of importances: {importance.sum():.4f}")
    print(f"Max importance: {importance.max():.4f}")
    print(f"Min importance: {importance.min():.4f}")
    print(f"Number of features with non-zero importance: {(importance > 0).sum()}")

    # Check that importance is non-negative
    assert np.all(importance >= 0), "Feature importances should be non-negative"

    # Check that at least some features have non-zero importance
    assert np.any(importance > 0), "At least some features should have non-zero importance"

    # Check that importances are normalized (sum to 1.0)
    assert np.abs(importance.sum() - 1.0) < 1e-6, f"Feature importances should sum to 1.0, got {importance.sum()}"

    print("✓ Single tree feature importance test passed!\n")
    return True


def test_forest_feature_importance():
    """Test feature importance for a Random Forest of PILOT trees."""
    print("=" * 80)
    print("Testing random forest feature importance")
    print("=" * 80)

    # Create a dataset
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a random forest
    rf = RandomForestCPilot(
        n_estimators=10,
        max_depth=5,
        n_features_tree=1.0,
        n_features_node=1.0,
        random_state=42
    )

    rf.fit(X_train, y_train)

    # Get feature importance
    importance = rf.feature_importances_

    # Print tree summaries
    print("\nTree Summaries:")
    for i, estimator in enumerate(rf.estimators):
        print(f"\n--- Tree {i+1} ---")
        print(estimator.tree_summary())

    print(f"Feature importances shape: {importance.shape}")
    print(f"Feature importances:\n{importance}")
    print(f"Sum of importances: {importance.sum():.4f}")
    print(f"Max importance: {importance.max():.4f}")
    print(f"Min importance: {importance.min():.4f}")
    print(f"Number of features with non-zero importance: {(importance > 0).sum()}")

    # Rank features by importance
    feature_ranking = np.argsort(importance)[::-1]
    print("\nFeature ranking (most to least important):")
    for rank, idx in enumerate(feature_ranking[:5], 1):
        print(f"  {rank}. Feature {idx}: {importance[idx]:.4f}")

    # Check that importance is non-negative
    assert np.all(importance >= 0), "Feature importances should be non-negative"

    # Check that importance has correct shape
    assert importance.shape[0] == X.shape[1], "Importance should have same length as number of features"

    # Check that at least some features have non-zero importance
    assert np.any(importance > 0), "At least some features should have non-zero importance"

    # Check that importances are normalized (sum to 1.0)
    assert np.abs(importance.sum() - 1.0) < 1e-6, f"Feature importances should sum to 1.0, got {importance.sum()}"

    # Make predictions to verify the model still works
    y_pred = rf.predict(X_test)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    print(f"\nModel R² on test set: {r2:.4f}")

    print("✓ Random forest feature importance test passed!\n")
    return True


def test_feature_subset():
    """Test that feature importance works correctly with feature subset selection."""
    print("=" * 80)
    print("Testing feature importance with feature subset")
    print("=" * 80)

    # Create a dataset
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=8,
        noise=10,
        random_state=42
    )

    # Train a random forest with subset of features per tree
    rf = RandomForestCPilot(
        n_estimators=10,
        max_depth=5,
        n_features_tree=0.5,  # Use only 50% of features per tree
        n_features_node=0.3,   # Use only 30% of features per node
        random_state=42
    )

    rf.fit(X, y)

    # Get feature importance
    importance = rf.feature_importances_

    # Print tree summaries
    print("\nTree Summaries:")
    for i, estimator in enumerate(rf.estimators):
        print(f"\n--- Tree {i+1} ---")
        print(estimator.tree_summary())

    print(f"Feature importances shape: {importance.shape}")
    print(f"Number of features in dataset: {X.shape[1]}")
    print(f"Number of features with non-zero importance: {(importance > 0).sum()}")

    # Check that importance array has correct size
    assert importance.shape[0] == X.shape[1], \
        f"Expected {X.shape[1]} importances, got {importance.shape[0]}"

    # Since we're using feature subset, not all features will be used
    # but some should definitely be used
    n_nonzero = (importance > 0).sum()
    print(f"Features used: {n_nonzero}/{X.shape[1]}")
    assert n_nonzero > 0, "At least some features should be used"

    print("✓ Feature subset test passed!\n")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PILOT Feature Importance Tests")
    print("="*80 + "\n")

    try:
        test_single_tree_feature_importance()
        test_forest_feature_importance()
        test_feature_subset()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
