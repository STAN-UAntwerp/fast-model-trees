"""
Comparison test to demonstrate the importance of normalization in feature importance.
This script shows why sklearn's normalization approach is important.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from pilot.c_ensemble import RandomForestCPilot

def compare_with_sklearn():
    """Compare PILOT feature importances with sklearn RandomForest."""
    print("=" * 80)
    print("Comparing PILOT with sklearn RandomForest feature importances")
    print("=" * 80)

    # Create a dataset
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )

    # Train sklearn RandomForest
    print("\nTraining sklearn RandomForestRegressor...")
    rf_sklearn = RandomForestRegressor(
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    rf_sklearn.fit(X, y)

    # Train PILOT RandomForest
    print("Training PILOT RandomForest...")
    rf_pilot = RandomForestCPilot(
        n_estimators=10,
        max_depth=5,
        n_features_tree=1.0,
        n_features_node=1.0,
        random_state=42
    )
    rf_pilot.fit(X, y)

    # Get feature importances
    importance_sklearn = rf_sklearn.feature_importances_
    importance_pilot = rf_pilot.feature_importances_

    # Print comparison
    print("\n" + "=" * 80)
    print("Feature Importance Comparison")
    print("=" * 80)
    print(f"{'Feature':<10} {'sklearn':<15} {'PILOT':<15} {'Rank (sklearn)':<15} {'Rank (PILOT)':<15}")
    print("-" * 80)

    rank_sklearn = np.argsort(importance_sklearn)[::-1]
    rank_pilot = np.argsort(importance_pilot)[::-1]

    for i in range(10):
        sklearn_rank = np.where(rank_sklearn == i)[0][0] + 1
        pilot_rank = np.where(rank_pilot == i)[0][0] + 1
        print(f"Feature {i:<3} {importance_sklearn[i]:<15.4f} {importance_pilot[i]:<15.4f} "
              f"{sklearn_rank:<15} {pilot_rank:<15}")

    print("-" * 80)
    print(f"{'Sum':<10} {importance_sklearn.sum():<15.4f} {importance_pilot.sum():<15.4f}")

    # Check that both are normalized
    print("\n" + "=" * 80)
    print("Normalization Check")
    print("=" * 80)
    print(f"sklearn importances sum: {importance_sklearn.sum():.10f}")
    print(f"PILOT importances sum:   {importance_pilot.sum():.10f}")
    print(f"Both sum to 1.0: {np.abs(importance_sklearn.sum() - 1.0) < 1e-6 and np.abs(importance_pilot.sum() - 1.0) < 1e-6}")

    # Compute rank correlation
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(importance_sklearn, importance_pilot)

    print("\n" + "=" * 80)
    print("Rank Correlation")
    print("=" * 80)
    print(f"Spearman correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")

    if correlation > 0.7:
        print("✓ High correlation between sklearn and PILOT feature importances!")
    elif correlation > 0.5:
        print("✓ Moderate correlation between sklearn and PILOT feature importances.")
    else:
        print("⚠ Low correlation - this is expected as PILOT uses linear models, not constant splits.")

    print("\n" + "=" * 80)
    print("✓ Comparison test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Feature Importance Comparison Test: PILOT vs sklearn")
    print("="*80 + "\n")

    try:
        compare_with_sklearn()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
