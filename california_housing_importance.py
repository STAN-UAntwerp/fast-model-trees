import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from pilot.c_ensemble import RandomForestCPilot


def main():
    # Load California Housing dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # Create a dataframe for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y

    # Create output directory for plots
    output_dir = os.path.join("Output", "california_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(output_dir)}")

    # Generate and save scatter plots
    print("Generating scatter plots...")
    for feature in feature_names:
        plt.figure(figsize=(10, 6))

        # Check for log scale
        is_log = feature in []
        if is_log:
            plt.xscale("log")

        plt.scatter(df[feature], df["Target"], alpha=0.5, s=10)

        xlabel = f"{feature} (log scale)" if is_log else feature
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel("Median House Value (x$100k)", fontsize=20)
        plt.tick_params(axis="both", labelsize=18)
        plt.grid(True, linestyle="--", alpha=0.7, which="both" if is_log else "major")

        # Save as PNG
        png_path = os.path.join(output_dir, f"scatter_{feature}.png")
        plt.savefig(png_path)

        # Save as PDF
        pdf_path = os.path.join(output_dir, f"scatter_{feature}.pdf")
        plt.savefig(pdf_path)

        plt.close()
        print(f"Saved {png_path} and {pdf_path}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RaFFLE
    rf_raffle = RandomForestCPilot(
        n_estimators=5,
        max_depth=5,
        n_features_tree=1.0,
        n_features_node=1.0,
        random_state=42,
        alpha=0.5,
    )

    print("\nTraining RandomForestCPilot...")
    rf_raffle.fit(X_train, y_train)
    print("Training complete.")

    # Get and print feature importances
    importances = rf_raffle.feature_importances_

    print("\nFeature Importances:")
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance:.4f}")

    # Print tree summaries
    print("\nTree Summaries:")
    for i, estimator in enumerate(rf_raffle.estimators):
        print(f"\n--- Tree {i+1} ---")
        summary = estimator.tree_summary(feature_names=feature_names)
        # Filter out constant nodes for cleaner output
        summary = summary[summary["node_type"] != "con"]
        print(summary.to_string())


if __name__ == "__main__":
    main()
