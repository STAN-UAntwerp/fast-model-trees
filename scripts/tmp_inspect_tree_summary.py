"""Temporary script to inspect the tree_summary output used in the new tests.

Run with: uv run python scripts/tmp_inspect_tree_summary.py
Not meant to be committed.
"""
import numpy as np
import pandas as pd
from pilot import PILOT, RaFFLE

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def make_categorical_data():
    """Same data as the `categorical_data` fixture in tests/test_pilot.py."""
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


def print_tree(df):
    """Print the tree as an indented outline, rebuilt from the summary only."""
    children = {}
    for _, row in df.iloc[1:].iterrows():  # left child is listed before right child
        children.setdefault(row["parent_node_id"], []).append(row)

    def describe(node):
        name = node.get("feature_name", f"x{node['feature_index']:.0f}")
        if node["node_type"] == "con":
            return f"con: {node['intercept_left']:+.3f}"
        if node["node_type"] == "lin":
            return f"lin: {node['intercept_left']:+.3f} {node['slope_left']:+.3f} * {name}"
        if node["node_type"] == "pconc":
            return (
                f"pconc: {name} in {node['pivot_values'].astype(int).tolist()} "
                f"-> {node['intercept_left']:+.3f}, else {node['intercept_right']:+.3f}"
            )
        return (
            f"{node['node_type']}: {name} <= {node['split_value']:.3f} "
            f"-> {node['intercept_left']:+.3f} {node['slope_left']:+.3f} * {name}, "
            f"else {node['intercept_right']:+.3f} {node['slope_right']:+.3f} * {name}"
        )

    def walk(node, indent, label):
        print(f"{indent}{label}[{node['node_id']}] {describe(node)}")
        kids = children.get(node["node_id"], [])
        labels = [""] if len(kids) == 1 else ["L ", "R "]
        for kid, kid_label in zip(kids, labels):
            walk(kid, indent + "    ", kid_label)

    walk(df.iloc[0], "", "")


def main():
    X, y, categorical = make_categorical_data()
    feature_names = ["num_a", "cat_b", "num_c"]

    model = PILOT(max_depth=4)
    model.train(X, y, categorical=categorical)
    df = model.tree_summary(feature_names=feature_names)

    print("=== PILOT.tree_summary() ===")
    print(df)

    print("\n=== pconc nodes: pivot values (levels going left) ===")
    print(
        df.loc[
            df["node_type"] == "pconc",
            ["node_id", "feature_name", "pivot_values", "intercept_left", "intercept_right"],
        ]
    )

    print("\n=== parent links: parent_node_id -> node_id ===")
    print(df[["model_depth", "node_id", "parent_node_id", "node_type"]])

    print("\n=== tree rebuilt from the summary, nodes shown as [node_id] ===")
    print_tree(df)

    print("\n=== RaFFLE: summary of the first tree ===")
    forest = RaFFLE(n_estimators=3, max_depth=4)
    forest.fit(X, y, categorical_idx=np.array([1]))
    tree = forest.estimators[0]
    print(f"feature_idx of this tree: {list(tree.feature_idx)}")
    print(tree.tree_summary(feature_names=feature_names))


if __name__ == "__main__":
    main()
