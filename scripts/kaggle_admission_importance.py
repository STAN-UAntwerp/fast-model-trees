import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from pilot.c_ensemble import RandomForestCPilot


def main():
    # Download dataset from GitHub (public repository with the same Kaggle dataset)
    url = "https://raw.githubusercontent.com/divyansha1115/Graduate-Admission-Prediction/master/Admission_Predict.csv"

    print("Downloading Graduate Admission dataset...")
    try:
        with urllib.request.urlopen(url) as response:
            data = pd.read_csv(response)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have internet connection or download manually from:")
        print("https://www.kaggle.com/datasets/adepvenugopal/graduate-admission-data")
        return

    # Prepare data
    # Drop 'Serial No.' column if it exists
    if "Serial No." in data.columns:
        data = data.drop("Serial No.", axis=1)

    # Define feature names and target
    feature_names = [col for col in data.columns if col != "Chance of Admit "]
    X = data[feature_names].values
    y = data["Chance of Admit "].values

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
        summary = summary[summary["node_type"] != "con"]
        print(summary.to_string())


if __name__ == "__main__":
    main()
