import click
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

outputfolder = pathlib.Path(__file__).parent.resolve() / "Output"
figurefolder = outputfolder / "paperplots"

MODELORDER = ["CART", "PILOT", "RF", "RaFFLE", "dRaFFLE", "XGB", "Lasso", "Ridge"]
MODELMAP = {
    "CPILOT": "PILOT",
    "CPF": "RaFFLE",
}
DRAFFLE = "CPF - df alpha = 0.5, no blin - max_depth = 20 - max_node_features = 1.0 - n_estimators = 100"


def load_basetable():
    # Load main results (RF, CPF, XGB, old CART/PILOT)
    basetable = pd.concat(
        [
            pd.read_csv(outputfolder / "cpilot_forest_benchmark_v11" / "results.csv"),
            pd.read_csv(
                outputfolder
                / "cpilot_forest_benchmark_v11_linear_models3"
                / "results.csv"
            ),
        ]
    )

    # Load new CART/PILOT HP tuning results
    cart_pilot_results = pd.read_csv(
        outputfolder / "cart_pilot_hp_tuning" / "results.csv"
    )

    # Remove old CART and CPILOT results from basetable
    basetable = basetable[~basetable["model"].str.startswith("CART")]
    basetable = basetable[~basetable["model"].str.startswith("CPILOT")]

    # Combine all results
    basetable = pd.concat([basetable, cart_pilot_results], ignore_index=True)

    # Average across folds for each HP combination
    basetable = basetable.groupby(["id", "model"])["r2"].mean().reset_index()
    basetable = basetable.assign(
        basemodel=basetable["model"].str.split("-").str[0].str.strip()
    )

    # Get best HP combination per base model
    besttable = (
        basetable.groupby(["id", "basemodel"])["r2"]
        .max()
        .reset_index()
        .rename(columns={"basemodel": "model"})
        .assign(model=lambda df: df["model"].map(lambda m: MODELMAP.get(m, m)))
    )

    # Get dRaFFLE specific configuration
    draffletable = (
        basetable.loc[basetable["model"] == DRAFFLE]
        .assign(model="dRaFFLE")
        .drop(columns="basemodel")
    )

    return pd.concat([besttable, draffletable])


def load_fit_duration_table():
    """Load fit_duration data from results files."""
    # Load main results (RF, CPF, XGB, old CART/PILOT)
    basetable = pd.concat(
        [
            pd.read_csv(outputfolder / "cpilot_forest_benchmark_v11" / "results.csv"),
            pd.read_csv(
                outputfolder
                / "cpilot_forest_benchmark_v11_linear_models3"
                / "results.csv"
            ),
        ]
    )

    # Load new CART/PILOT HP tuning results
    cart_pilot_results = pd.read_csv(
        outputfolder / "cart_pilot_hp_tuning" / "results.csv"
    )

    # Remove old CART and CPILOT results from basetable
    basetable = basetable[~basetable["model"].str.startswith("CART")]
    basetable = basetable[~basetable["model"].str.startswith("CPILOT")]

    # Combine all results
    basetable = pd.concat([basetable, cart_pilot_results], ignore_index=True)

    # Average across folds for each HP combination
    basetable = basetable.groupby(["id", "model"])["fit_duration"].mean().reset_index()
    basetable = basetable.assign(
        basemodel=basetable["model"].str.split("-").str[0].str.strip()
    )

    # For each basemodel, select the configuration with the best (lowest) fit_duration
    besttable = (
        basetable.groupby(["id", "basemodel"])["fit_duration"]
        .min()
        .reset_index()
        .rename(columns={"basemodel": "model"})
        .assign(model=lambda df: df["model"].map(lambda m: MODELMAP.get(m, m)))
    )
    draffletable = (
        basetable.loc[basetable["model"] == DRAFFLE]
        .assign(model="dRaFFLE")
        .drop(columns="basemodel")
    )

    return pd.concat([besttable, draffletable])


def get_transformation_table():
    original_results = pd.concat(
        [
            pd.read_csv(outputfolder / "cpilot_forest_benchmark_v11/results.csv"),
            pd.read_csv(
                outputfolder / "cpilot_forest_benchmark_v11_linear_models3/results.csv"
            ),
        ],
        axis=0,
    )

    new_results = pd.read_csv(outputfolder / "benchmark_power_transform_v2/results.csv")

    original_results = (
        original_results.groupby(["id", "model"])["r2"].mean().reset_index()
    )
    new_results = new_results.groupby(["id", "model"])["r2"].mean().reset_index()

    original_best = (
        original_results.assign(
            model=original_results["model"].str.split("-").str[0].str.strip()
        )
        .groupby(["id", "model"])["r2"]
        .max()
        .reset_index()
    )
    new_best = (
        new_results.assign(model=new_results["model"].str.split("-").str[0].str.strip())
        .groupby(["id", "model"])["r2"]
        .max()
        .reset_index()
    )

    df = pd.merge(
        left=original_best,
        right=new_best,
        on=["id", "model"],
        suffixes=["_original", "_transformed"],
        how="inner",
    )

    df = df.assign(r2_delta=(df["r2_transformed"] - df["r2_original"]).clip(-1, 1))
    df["model"] = df["model"].map(lambda m: MODELMAP.get(m, m))
    return df


def get_relative_table(basetable):
    reltable = basetable.pivot(index="id", columns="model", values="r2")
    lintype = pd.Series(
        index=reltable.index,
        data=np.where(
            reltable[["Lasso", "Ridge"]].max(axis=1) > reltable["CART"],
            "Linear",
            "Non-linear",
        ).flatten(),
        name="Type",
    )
    reltable = reltable.clip(0, 1) / reltable.clip(0, 1).max(axis=1).values.reshape(
        -1, 1
    )
    return (
        reltable.reset_index()
        .melt(id_vars="id", var_name="model", value_name="r2")
        .set_index("id")
        .join(lintype)
        .reset_index()
    )


def plot_overall_boxplot(reltable):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.boxplot(data=reltable, x="model", y="r2", ax=ax, order=MODELORDER)
    ax.set_ylabel(r"Relative $R^2$", fontsize=30)
    ax.set_xlabel(None)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    fig.tight_layout()
    fig.savefig(outputfolder / "paperplots" / "boxplots_overall_relative.png", dpi=300)
    fig.savefig(outputfolder / "paperplots" / "boxplots_overall_relative.pdf", dpi=300)


def plot_lin_vs_nonlin_boxplot(reltable):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.boxplot(data=reltable, x="model", y="r2", hue="Type", ax=ax, order=MODELORDER)
    ax.set_ylabel(r"Relative $R^2$", fontsize=30)
    ax.set_xlabel(None)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    _ = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=26)
    fig.tight_layout()
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_lin_vs_nonlin_relative_all.png", dpi=300
    )
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_lin_vs_nonlin_relative_all.pdf", dpi=300
    )


def plot_delta_transform(transformtable):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    df = transformtable[
        transformtable["model"].isin(["PILOT", "RaFFLE", "Lasso", "Ridge"])
    ]
    sns.boxplot(df, x="model", y="r2_delta", ax=ax)
    ax.set_ylabel(r"Delta $R^2$", fontsize=30)
    ax.set_xlabel("Model", fontsize=30)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    fig.tight_layout()
    fig.savefig(outputfolder / "paperplots" / "boxplots_transform.png", dpi=300)
    fig.savefig(outputfolder / "paperplots" / "boxplots_transform.pdf", dpi=300)


def plot_linear_convergence():
    df = pd.read_csv(outputfolder / "linear_experiment_v3" / "results.csv")
    df = (
        df.groupby(["noise", "n_samples", "model"])["r2"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "noise": "noise standard deviation",
                "n_samples": "Training set size",
                "model": "Method",
            }
        )
    )

    g = sns.FacetGrid(
        df,
        col="noise standard deviation",
        hue="Method",
        height=4,
        aspect=1.5,
        col_wrap=2,
    )

    # Add line plots with markers
    g.map(sns.lineplot, "Training set size", "r2", marker="o")

    # Add reference lines (adjust these based on actual data)
    for ax, (noise, subset) in zip(
        g.axes.flat, df.groupby(["noise standard deviation"])
    ):
        subset = subset.pivot(columns="Method", index="Training set size", values="r2")
        x1 = subset[subset["CPF"] > 0.97 * subset["LR"]].index.min()
        x2 = subset[subset["CPF"] > 0.99 * subset["LR"]].index.min()
        ax.axvline(
            x=x1, linestyle=":", color="black", label="RaFFLE reaches 97% of OLS"
        )
        ax.axvline(
            x=x2, linestyle="--", color="black", label="RaFFLE reaches 99% of OLS"
        )

    # Adjust labels
    g.set_axis_labels("Number of Samples", "Average $R^2$")
    g.add_legend(title="Method")

    plt.savefig(figurefolder / "linear_convergence.png")


def plot_linear_convergence_high_noise():
    df = pd.read_csv(outputfolder / "linear_experiment_high_noise" / "results.csv")

    # Apply MODELMAP to rename models
    df["model"] = df["model"].map(lambda m: MODELMAP.get(m, m))

    df = (
        df.groupby(["noise", "n_samples", "model"])["r2"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "noise": "noise standard deviation",
                "n_samples": "Training set size",
                "model": "Method",
            }
        )
    )

    # Create a single plot instead of FacetGrid
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Plot lines for each method
    sns.lineplot(
        data=df, x="Training set size", y="r2", hue="Method", marker="o", ax=ax
    )

    # Add reference line where RaFFLE flattens off
    pivot_df = df.pivot(columns="Method", index="Training set size", values="r2")
    if "RaFFLE" in pivot_df.columns and "LR" in pivot_df.columns:
        # Calculate the improvement rate (difference between consecutive points)
        raffle_values = pivot_df["RaFFLE"].dropna()
        improvements = raffle_values.diff()

        # Find where improvement becomes very small (< 1% of the range)
        threshold = 0.001 * (raffle_values.max() - raffle_values.min())
        flatten_idx = improvements[improvements.abs() < threshold].index

        if len(flatten_idx) > 0:
            # Take the first point where it flattens
            flatten_point = flatten_idx[0]
            raffle_r2 = pivot_df.loc[flatten_point, "RaFFLE"]
            lr_r2 = pivot_df.loc[flatten_point, "LR"]
            percentage = (raffle_r2 / lr_r2) * 100

            ax.axvline(
                x=flatten_point,
                linestyle="--",
                color="black",
                label=f"RaFFLE flattens off ({percentage:.1f}% of OLS)",
            )

    # Get noise level for title
    noise_level = df["noise standard deviation"].iloc[0]
    ax.set_title(f"noise standard deviation = {noise_level}")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Average $R^2$")
    ax.legend(title="Method")

    fig.tight_layout()
    fig.savefig(
        figurefolder / "linear_convergence_high_noise.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        figurefolder / "linear_convergence_high_noise.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_fit_duration_boxplot(fit_duration_table):
    """Plot boxplot with absolute fit_duration values per model."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.boxplot(
        data=fit_duration_table, x="model", y="fit_duration", ax=ax, order=MODELORDER
    )
    ax.set_ylabel("Fit Duration (s)", fontsize=30)
    ax.set_xlabel("Model", fontsize=30)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    fig.tight_layout()
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_fit_duration_absolute.png", dpi=300
    )
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_fit_duration_absolute.pdf", dpi=300
    )


def plot_fit_duration_boxplot_log(fit_duration_table):
    """Plot boxplot with absolute fit_duration values per model (log scale)."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.boxplot(
        data=fit_duration_table, x="model", y="fit_duration", ax=ax, order=MODELORDER
    )
    ax.set_yscale("log")
    ax.set_ylabel("Fit Duration (s, log scale)", fontsize=30)
    ax.set_xlabel("Model", fontsize=30)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    fig.tight_layout()
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_fit_duration_absolute_log.png", dpi=300
    )
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_fit_duration_absolute_log.pdf", dpi=300
    )


def get_relative_fit_duration_table(fit_duration_table):
    """Calculate relative fit_duration by dividing by the fastest model per dataset."""
    reltable = fit_duration_table.pivot(
        index="id", columns="model", values="fit_duration"
    )

    # Find the minimum (fastest) time across models for each dataset
    min_time = reltable.min(axis=1).values.reshape(-1, 1)

    # Divide all times by the minimum time
    reltable = reltable / min_time

    return reltable.reset_index().melt(
        id_vars="id", var_name="model", value_name="fit_duration"
    )


def plot_relative_fit_duration_boxplot(rel_fit_duration_table):
    """Plot boxplot with relative fit_duration values per model."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.boxplot(
        data=rel_fit_duration_table,
        x="model",
        y="fit_duration",
        ax=ax,
        order=MODELORDER,
    )
    ax.set_ylabel("Relative Fit Duration", fontsize=30)
    ax.set_xlabel("Model", fontsize=30)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    fig.tight_layout()
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_fit_duration_relative.png", dpi=300
    )
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_fit_duration_relative.pdf", dpi=300
    )


def plot_relative_fit_duration_boxplot_log(rel_fit_duration_table):
    """Plot boxplot with relative fit_duration values per model (log scale)."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.boxplot(
        data=rel_fit_duration_table,
        x="model",
        y="fit_duration",
        ax=ax,
        order=MODELORDER,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Relative Fit Duration (log scale)", fontsize=30)
    ax.set_xlabel("Model", fontsize=30)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    fig.tight_layout()
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_fit_duration_relative_log.png", dpi=300
    )
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_fit_duration_relative_log.pdf", dpi=300
    )


def create_fit_duration_latex_table(fit_duration_table):
    """Create LaTeX table with absolute fit_duration values per model per dataset."""
    # Pivot the table to have datasets as rows and models as columns
    table = fit_duration_table.pivot(index="id", columns="model", values="fit_duration")

    # Reorder columns according to MODELORDER
    table = table[[col for col in MODELORDER if col in table.columns]]

    # Format values to 3 decimal places
    table = table.map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")

    # Generate LaTeX
    latex_str = table.to_latex(
        escape=False,
        column_format="l" + "c" * len(table.columns),
        caption="Fit duration (in seconds) per model per dataset",
        label="tab:fit_duration",
    )

    # Save to file
    with open(outputfolder / "paperplots" / "fit_duration_table.tex", "w") as f:
        f.write(latex_str)

    print(
        f"LaTeX table saved to {outputfolder / 'paperplots' / 'fit_duration_table.tex'}"
    )


def create_relative_r2_latex_table(basetable):
    """Create LaTeX tables with relative R² values (with clipping) per model per dataset.
    Splits the table across multiple pages following the paper format."""
    # Load the source information if available
    try:
        # Try to get source column from one of the results files
        results_df = pd.read_csv(
            outputfolder / "cpilot_forest_benchmark_v11" / "results.csv"
        )
        source_map = (
            results_df[["id", "source"]]
            .drop_duplicates()
            .set_index("id")["source"]
            .to_dict()
        )
    except:
        source_map = {}

    # Create relative table with clipping (original behavior)
    reltable = basetable.pivot(index="id", columns="model", values="r2")
    reltable_clipped = reltable.clip(0, 1)
    reltable_normalized = reltable_clipped / reltable_clipped.max(
        axis=1
    ).values.reshape(-1, 1)

    # Reorder columns according to MODELORDER
    reltable_normalized = reltable_normalized[
        [col for col in MODELORDER if col in reltable_normalized.columns]
    ]

    # Filter out uci_162 (dataset with all negative R² values)
    reltable_normalized = reltable_normalized[reltable_normalized.index != "uci_162"]

    # Add source column
    reltable_normalized.insert(
        0, "source", reltable_normalized.index.map(lambda x: source_map.get(x, ""))
    )
    reltable_normalized = reltable_normalized.reset_index()

    # Sort by source (PMLB before UCI) and then by numeric part of id
    def extract_sort_keys(id_str):
        parts = str(id_str).split("_", 1)
        if len(parts) == 2:
            source = parts[0].lower()
            try:
                numeric_id = int(parts[1])
            except ValueError:
                numeric_id = 0
            return (source, numeric_id)
        return ("", 0)

    reltable_normalized["_sort_keys"] = reltable_normalized["id"].apply(
        extract_sort_keys
    )
    reltable_normalized = reltable_normalized.sort_values("_sort_keys").drop(
        columns="_sort_keys"
    )

    # Split data into roughly 3 equal parts
    n_rows = len(reltable_normalized)
    split1 = n_rows // 3
    split2 = 2 * n_rows // 3

    tables = [
        reltable_normalized.iloc[:split1],
        reltable_normalized.iloc[split1:split2],
        reltable_normalized.iloc[split2:],
    ]

    # Calculate average and std for the footer (across all datasets)
    avg_row = reltable_normalized[
        [col for col in MODELORDER if col in reltable_normalized.columns]
    ].mean()
    std_row = reltable_normalized[
        [col for col in MODELORDER if col in reltable_normalized.columns]
    ].std()

    latex_output = ""

    for i, table_part in enumerate(tables, 1):
        latex_output += f"\\renewcommand{{\\arraystretch}}{{0.9}}\n"
        latex_output += f"\\begin{{table}}[ht]\n"
        latex_output += f"\\caption{{Average $R^2$ score divided by highest average $R^2$ score per dataset ({i}/III)}}\n"
        latex_output += f"\\label{{tab:empirical_results_p{i}}}\n"
        latex_output += f"\\centering\n"
        latex_output += f"\\footnotesize\n"

        # Build column format
        model_cols = [col for col in MODELORDER if col in table_part.columns]
        n_cols = len(model_cols) + 2  # +2 for source and id
        latex_output += (
            f"\\begin{{tabularx}}{{\\textwidth}}{{ll" + "r" * len(model_cols) + "}\n"
        )
        latex_output += "\\toprule\n"

        # Header row with special formatting for dRaFFLE
        header = "source &  id"
        for col in model_cols:
            if col == "dRaFFLE":
                header += " &  \\makecell{RaFFLE \\\\Default}"
            else:
                header += f" & {col:>6}"
        header += " \\\\\n"
        latex_output += header
        latex_output += "\\midrule\n"

        # Data rows
        for _, row in table_part.iterrows():
            # Find best value(s) in this row (within tolerance)
            row_values = {col: row[col] for col in model_cols}
            max_val = max(row_values.values())

            # Split id into source and numeric id
            id_parts = str(row["id"]).split("_", 1)
            if len(id_parts) == 2:
                source_part, id_part = id_parts
            else:
                source_part, id_part = "", str(row["id"])

            # Capitalize source (PMLB, UCI, etc.)
            source_part = source_part.upper()

            line = f"{source_part:>6} & {id_part:>3}"
            for col in model_cols:
                val = row[col]
                if pd.notna(val):
                    # Check if this is a best value (within 0.01 tolerance)
                    if val >= max_val - 0.01:
                        line += f" & \\textbf{{{val:>4.2f}}}"
                    else:
                        line += f" & {val:>6.2f}"
                else:
                    line += " &      -"
            line += " \\\\\n"
            latex_output += line

        # Add average and std in the last table
        if i == 3:
            latex_output += "\\midrule\n"
            line = "       & average"
            for col in model_cols:
                val = avg_row[col]
                if val >= avg_row.max() - 0.01:
                    line += f" & \\textbf{{{val:>4.2f}}}"
                else:
                    line += f" & {val:>10.2f}"
            line += " \\\\\n"
            latex_output += line

            line = "       &  std"
            for col in model_cols:
                line += f" & {std_row[col]:>10.2f}"
            line += " \\\\\n"
            latex_output += line

        latex_output += "\\bottomrule\n"
        latex_output += "\\end{tabularx}\n"
        latex_output += "\\end{table}\n\n"

    # Save to file
    with open(outputfolder / "paperplots" / "relative_r2_table_clipped.tex", "w") as f:
        f.write(latex_output)

    print(
        f"LaTeX table saved to {outputfolder / 'paperplots' / 'relative_r2_table_clipped.tex'}"
    )


def create_relative_r2_latex_table_unclipped(basetable):
    """Create LaTeX tables with relative R² values without clipping negative values.
    Values that differ from the clipped version are highlighted in red.
    Splits the table across multiple pages following the paper format."""
    # Load the source information if available
    try:
        results_df = pd.read_csv(
            outputfolder / "cpilot_forest_benchmark_v11" / "results.csv"
        )
        source_map = (
            results_df[["id", "source"]]
            .drop_duplicates()
            .set_index("id")["source"]
            .to_dict()
        )
    except:
        source_map = {}

    # Create relative table WITH clipping (for comparison)
    reltable = basetable.pivot(index="id", columns="model", values="r2")
    reltable_clipped = reltable.clip(0, 1)
    reltable_clipped_normalized = reltable_clipped / reltable_clipped.max(
        axis=1
    ).values.reshape(-1, 1)

    # Create relative table WITHOUT clipping
    # Use the same normalization factor (max of clipped values) for consistency
    max_clipped = reltable_clipped.max(axis=1).values.reshape(-1, 1)
    # Avoid division by zero - if max is 0, set it to 1 to avoid inf
    max_clipped = np.where(max_clipped == 0, 1, max_clipped)
    reltable_unclipped_normalized = reltable / max_clipped

    # Reorder columns according to MODELORDER
    model_cols = [
        col for col in MODELORDER if col in reltable_unclipped_normalized.columns
    ]
    reltable_clipped_normalized = reltable_clipped_normalized[model_cols]
    reltable_unclipped_normalized = reltable_unclipped_normalized[model_cols]

    # Filter out uci_162 (dataset with all negative R² values)
    reltable_clipped_normalized = reltable_clipped_normalized[
        reltable_clipped_normalized.index != "uci_162"
    ]
    reltable_unclipped_normalized = reltable_unclipped_normalized[
        reltable_unclipped_normalized.index != "uci_162"
    ]

    # Add source column
    reltable_unclipped_normalized.insert(
        0,
        "source",
        reltable_unclipped_normalized.index.map(lambda x: source_map.get(x, "")),
    )
    reltable_unclipped_normalized = reltable_unclipped_normalized.reset_index()

    reltable_clipped_normalized.insert(
        0,
        "source",
        reltable_clipped_normalized.index.map(lambda x: source_map.get(x, "")),
    )
    reltable_clipped_normalized = reltable_clipped_normalized.reset_index()

    # Sort by source (PMLB before UCI) and then by numeric part of id
    def extract_sort_keys(id_str):
        parts = str(id_str).split("_", 1)
        if len(parts) == 2:
            source = parts[0].lower()
            try:
                numeric_id = int(parts[1])
            except ValueError:
                numeric_id = 0
            return (source, numeric_id)
        return ("", 0)

    reltable_unclipped_normalized["_sort_keys"] = reltable_unclipped_normalized[
        "id"
    ].apply(extract_sort_keys)
    reltable_unclipped_normalized = (
        reltable_unclipped_normalized.sort_values("_sort_keys")
        .drop(columns="_sort_keys")
        .reset_index(drop=True)
    )

    reltable_clipped_normalized["_sort_keys"] = reltable_clipped_normalized["id"].apply(
        extract_sort_keys
    )
    reltable_clipped_normalized = (
        reltable_clipped_normalized.sort_values("_sort_keys")
        .drop(columns="_sort_keys")
        .reset_index(drop=True)
    )

    # Split data into roughly 3 equal parts
    n_rows = len(reltable_unclipped_normalized)
    split1 = n_rows // 3
    split2 = 2 * n_rows // 3

    tables_unclipped = [
        reltable_unclipped_normalized.iloc[:split1],
        reltable_unclipped_normalized.iloc[split1:split2],
        reltable_unclipped_normalized.iloc[split2:],
    ]

    tables_clipped = [
        reltable_clipped_normalized.iloc[:split1],
        reltable_clipped_normalized.iloc[split1:split2],
        reltable_clipped_normalized.iloc[split2:],
    ]

    # Calculate average and std for the footer (across all datasets)
    avg_row = reltable_unclipped_normalized[model_cols].mean()
    std_row = reltable_unclipped_normalized[model_cols].std()

    latex_output = "% Requires \\usepackage{xcolor} in your LaTeX preamble\n\n"

    for table_idx, (table_unclipped, table_clipped) in enumerate(
        zip(tables_unclipped, tables_clipped), 1
    ):
        latex_output += f"\\renewcommand{{\\arraystretch}}{{0.9}}\n"
        latex_output += f"\\begin{{table}}[ht]\n"
        latex_output += f"\\caption{{Average $R^2$ score divided by highest average $R^2$ score per dataset (without clipping) ({table_idx}/III). Values in \\textcolor{{red}}{{red}} differ from the clipped version.}}\n"
        latex_output += f"\\label{{tab:empirical_results_unclipped_p{table_idx}}}\n"
        latex_output += f"\\centering\n"
        latex_output += f"\\footnotesize\n"

        # Build column format
        n_cols = len(model_cols) + 2  # +2 for source and id
        latex_output += (
            f"\\begin{{tabularx}}{{\\textwidth}}{{ll" + "r" * len(model_cols) + "}\n"
        )
        latex_output += "\\toprule\n"

        # Header row with special formatting for dRaFFLE
        header = "source &  id"
        for col in model_cols:
            if col == "dRaFFLE":
                header += " &  \\makecell{RaFFLE \\\\Default}"
            else:
                header += f" & {col:>6}"
        header += " \\\\\n"
        latex_output += header
        latex_output += "\\midrule\n"

        # Data rows
        for row_idx, (idx_unclipped, row_unclipped) in enumerate(
            table_unclipped.iterrows()
        ):
            row_clipped = table_clipped.iloc[row_idx]

            # Find best value(s) in this row (within tolerance)
            row_values = {col: row_unclipped[col] for col in model_cols}
            max_val = max(row_values.values())

            # Split id into source and numeric id
            id_parts = str(row_unclipped["id"]).split("_", 1)
            if len(id_parts) == 2:
                source_part, id_part = id_parts
            else:
                source_part, id_part = "", str(row_unclipped["id"])

            # Capitalize source (PMLB, UCI, etc.)
            source_part = source_part.upper()

            line = f"{source_part:>6} & {id_part:>3}"
            for col in model_cols:
                val_unclipped = row_unclipped[col]
                val_clipped = row_clipped[col]

                if pd.notna(val_unclipped):
                    # Check if values differ (with small tolerance for floating point comparison)
                    differs = abs(val_clipped - val_unclipped) > 1e-6

                    # Check if this is a best value (within 0.01 tolerance)
                    is_best = val_unclipped >= max_val - 0.01

                    if differs:
                        if is_best:
                            line += f" & \\textcolor{{red}}{{\\textbf{{{val_unclipped:>4.2f}}}}}"
                        else:
                            line += f" & \\textcolor{{red}}{{{val_unclipped:>6.2f}}}"
                    else:
                        if is_best:
                            line += f" & \\textbf{{{val_unclipped:>4.2f}}}"
                        else:
                            line += f" & {val_unclipped:>6.2f}"
                else:
                    line += " &      -"
            line += " \\\\\n"
            latex_output += line

        # Add average and std in the last table
        if table_idx == 3:
            latex_output += "\\midrule\n"
            line = "       & average"
            for col in model_cols:
                val = avg_row[col]
                if val >= avg_row.max() - 0.01:
                    line += f" & \\textbf{{{val:>4.2f}}}"
                else:
                    line += f" & {val:>10.2f}"
            line += " \\\\\n"
            latex_output += line

            line = "       &  std"
            for col in model_cols:
                line += f" & {std_row[col]:>10.2f}"
            line += " \\\\\n"
            latex_output += line

        latex_output += "\\bottomrule\n"
        latex_output += "\\end{tabularx}\n"
        latex_output += "\\end{table}\n\n"

    # Save to file
    with open(
        outputfolder / "paperplots" / "relative_r2_table_unclipped.tex", "w"
    ) as f:
        f.write(latex_output)

    print(
        f"LaTeX table saved to {outputfolder / 'paperplots' / 'relative_r2_table_unclipped.tex'}"
    )


def create_r2_pairplot(basetable):
    """Create pairplots of raw R² values with clipping for CART, Lasso, and Ridge.

    Shows scatter plots with y=x line and histograms on diagonal.
    """
    # Pivot the table to have datasets as rows and models as columns
    r2_table = basetable.pivot(index="id", columns="model", values="r2")

    # Reorder columns according to MODELORDER
    model_cols = ["CART", "Lasso", "Ridge"]
    r2_table = r2_table[model_cols]

    # Clip values to [0, 1] for CART, Lasso, and Ridge
    for model in model_cols:
        if model in r2_table.columns:
            r2_table[model] = r2_table[model].clip(0, 1)

    # Filter out uci_162 (dataset with all negative R² values)
    r2_table = r2_table[r2_table.index != "uci_162"]

    # Create custom PairGrid for better control
    g = sns.PairGrid(r2_table, diag_sharey=False, height=2.5, aspect=1)

    # Map scatter plots to the lower triangle with y=x line
    def scatter_with_line(x, y, **kwargs):
        plt.scatter(x, y, alpha=0.5, s=20, **kwargs)
        # Get axis limits
        lims = [max(plt.xlim()[0], plt.ylim()[0]), min(plt.xlim()[1], plt.ylim()[1])]
        # Plot y=x line
        plt.plot(lims, lims, "r-", alpha=0.75, zorder=0, linewidth=1.5)
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())

    # Map histograms to diagonal
    g.map_diag(plt.hist, bins=20, edgecolor="black", alpha=0.7)

    # Map scatter plots with y=x line to off-diagonal
    g.map_offdiag(scatter_with_line)

    # Increase fontsize of axis labels
    for ax in g.axes.flatten():
        ax.tick_params(axis='both', labelsize=16)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)

    # Adjust layout
    plt.tight_layout()

    # Save figures
    g.savefig(figurefolder / "r2_pairplot.png", dpi=300, bbox_inches="tight")
    g.savefig(figurefolder / "r2_pairplot.pdf", dpi=300, bbox_inches="tight")

    print(f"R² pairplot saved to {figurefolder / 'r2_pairplot.png'}")
    print(f"R² pairplot saved to {figurefolder / 'r2_pairplot.pdf'}")


def compare_feature_importance(X, y, feature_names, dataset_name, file_prefix):
    """Compare feature importances between RaFFLE and sklearn RandomForest."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from pilot.c_ensemble import RandomForestCPilot
    import pickle

    # Check if cached results exist
    cache_file = figurefolder / f"{file_prefix}_cv_results.pkl"

    if cache_file.exists():
        print(f"Loading cached cross-validation results from {cache_file}...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)

        importance_sklearn = cached_data["importance_sklearn"]
        importance_raffle = cached_data["importance_raffle"]
        r2_sklearn = cached_data["r2_sklearn"]
        r2_sklearn_std = cached_data["r2_sklearn_std"]
        r2_raffle = cached_data["r2_raffle"]
        r2_raffle_std = cached_data["r2_raffle_std"]
        node_type_counts = cached_data.get("node_type_counts", None)

        print(
            f"Loaded results: sklearn R²={r2_sklearn:.4f}±{r2_sklearn_std:.4f}, RaFFLE R²={r2_raffle:.4f}±{r2_raffle_std:.4f}"
        )
    else:
        # Use 5-fold cross-validation
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Storage for feature importances and scores across folds
        importance_sklearn_folds = []
        importance_raffle_folds = []
        r2_sklearn_folds = []
        r2_raffle_folds = []

        print(f"Running {n_folds}-fold cross-validation on {dataset_name}...")
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            print(f"  Fold {fold_idx}/{n_folds}...")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train sklearn RandomForest
            rf_sklearn = RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            )
            rf_sklearn.fit(X_train, y_train)
            importance_sklearn_folds.append(rf_sklearn.feature_importances_)
            r2_sklearn_folds.append(rf_sklearn.score(X_test, y_test))

            # Train RaFFLE
            rf_raffle = RandomForestCPilot(
                n_estimators=100,
                max_depth=5,
                n_features_tree=1.0,
                n_features_node=1.0,
                random_state=42,
            )
            rf_raffle.fit(X_train, y_train)
            importance_raffle_folds.append(rf_raffle.feature_importances_)
            y_pred = rf_raffle.predict(X_test)
            r2_raffle = 1 - np.sum((y_test - y_pred) ** 2) / np.sum(
                (y_test - y_test.mean()) ** 2
            )
            r2_raffle_folds.append(r2_raffle)

        # Average feature importances across folds
        importance_sklearn = np.mean(importance_sklearn_folds, axis=0)
        importance_raffle = np.mean(importance_raffle_folds, axis=0)

        # Average R² scores
        r2_sklearn = np.mean(r2_sklearn_folds)
        r2_raffle = np.mean(r2_raffle_folds)
        r2_sklearn_std = np.std(r2_sklearn_folds)
        r2_raffle_std = np.std(r2_raffle_folds)

        # Analyze node types per feature for RaFFLE (using last fold's model)
        print(f"\nAnalyzing node types per feature for RaFFLE ({dataset_name})...")

        # Initialize counters for each node type per feature
        node_type_counts = {
            fname: {
                "con": 0,
                "lin": 0,
                "pcon": 0,
                "blin": 0,
                "plin": 0,
                "pconc": 0,
                "total": 0,
            }
            for fname in feature_names
        }

        # Aggregate node types across all trees in the last fold's forest
        for tree in rf_raffle.estimators:
            # tree_summary already handles feature_idx mapping internally
            tree_summary = tree.tree_summary(feature_names=feature_names)

            # Count node types per feature
            for _, row in tree_summary.iterrows():
                if not pd.isna(row.get("feature_name")):
                    node_type = row["node_type"]
                    feature = row["feature_name"]
                    if feature in node_type_counts:
                        node_type_counts[feature][node_type] += 1
                        node_type_counts[feature]["total"] += 1

        # Save results to cache
        print(f"Saving cross-validation results to {cache_file}...")
        cached_data = {
            "importance_sklearn": importance_sklearn,
            "importance_raffle": importance_raffle,
            "r2_sklearn": r2_sklearn,
            "r2_sklearn_std": r2_sklearn_std,
            "r2_raffle": r2_raffle,
            "r2_raffle_std": r2_raffle_std,
            "node_type_counts": node_type_counts,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f)
        print(f"Cached results saved.")

    # Create comparison dataframe
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "sklearn RF": importance_sklearn,
            "RaFFLE": importance_raffle,
            "Difference": importance_raffle - importance_sklearn,
        }
    )

    # Sort by absolute difference
    importance_df = (
        importance_df.assign(abs_diff=lambda df: np.abs(df["Difference"]))
        .sort_values("abs_diff", ascending=False)
        .drop("abs_diff", axis=1)
    )

    # Create dataframe for node type analysis (if available)
    if node_type_counts is not None:
        node_type_df = pd.DataFrame.from_dict(node_type_counts, orient="index")
        node_type_df = node_type_df.sort_values("total", ascending=False)

        # Save as markdown table
        node_type_filename = f"{file_prefix}_node_types.md"
        markdown_table = f"# RaFFLE Node Type Analysis ({dataset_name})\\n\\n"
        markdown_table += "Analysis based on 100 trees from the last fold.\\n\\n"
        markdown_table += "## Node Type Counts per Feature\\n\\n"
        markdown_table += node_type_df.to_markdown()
        markdown_table += "\\n\\n## Node Type Legend\\n\\n"
        markdown_table += "- **con**: Constant node (no split)\\n"
        markdown_table += "- **lin**: Linear node (simple linear model)\\n"
        markdown_table += "- **pcon**: Piecewise constant\\n"
        markdown_table += "- **blin**: Bilinear\\n"
        markdown_table += "- **plin**: Piecewise linear\\n"
        markdown_table += "- **pconc**: Piecewise constant constrained\\n"
        markdown_table += "- **total**: Total number of splits using this feature\\n"

        with open(figurefolder / node_type_filename, "w") as f:
            f.write(markdown_table)

        print(f"Node type analysis saved to {figurefolder / node_type_filename}")

    # Reorder for plotting (by sklearn importance)
    plot_order = np.argsort(importance_sklearn)[::-1]
    features_ordered = [feature_names[i] for i in plot_order]
    sklearn_ordered = importance_sklearn[plot_order]
    raffle_ordered = importance_raffle[plot_order]

    # Figure 1: Side-by-side comparison
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    x = np.arange(len(feature_names))
    width = 0.35

    ax1.bar(
        x - width / 2,
        sklearn_ordered,
        width,
        label="sklearn RF",
        color="#3274A1",
        alpha=0.8,
    )
    ax1.bar(
        x + width / 2, raffle_ordered, width, label="RaFFLE", color="#E1812C", alpha=0.8
    )

    ax1.set_xlabel("Features", fontsize=20)
    ax1.set_ylabel("Feature Importance", fontsize=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(features_ordered, rotation=45, ha="right", fontsize=18)
    ax1.tick_params(axis="y", labelsize=18)
    ax1.legend(fontsize=18)
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_filename1_pdf = f"{file_prefix}_comparison.pdf"
    plot_filename1_png = f"{file_prefix}_comparison.png"
    fig1.savefig(figurefolder / plot_filename1_pdf, dpi=300, bbox_inches="tight")
    fig1.savefig(figurefolder / plot_filename1_png, dpi=300, bbox_inches="tight")

    # Figure 2: Difference plot (RaFFLE - sklearn)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    differences = importance_df["Difference"].values
    colors = ["#2CA02C" if d > 0 else "#D62728" for d in differences]

    ax2.barh(importance_df["Feature"], differences, color=colors, alpha=0.7)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Importance Difference (RaFFLE - sklearn RF)", fontsize=20)
    ax2.set_ylabel("Features", fontsize=20)
    ax2.tick_params(axis="both", labelsize=18)
    ax2.grid(axis="x", alpha=0.3)

    # Add text annotations for significant differences
    # Place text inside bar to avoid overlap with y-axis labels
    for i, diff in enumerate(differences):
        if abs(diff) > 0.02:  # Only annotate significant differences
            # Place text inside the bar, away from the end
            if diff > 0:
                # For positive bars, place text near the right end but inside
                text_pos = diff * 0.7
                ha = "right"
            else:
                # For negative bars, place text near the left end but inside
                text_pos = diff * 0.7
                ha = "left"

            ax2.text(
                text_pos,
                i,
                f"{diff:+.3f}",
                va="center",
                ha=ha,
                fontsize=16,
                fontweight="bold",
                color="black",  # White text for better contrast on colored bars
            )

    plt.tight_layout()
    plot_filename2_pdf = f"{file_prefix}_differences.pdf"
    plot_filename2_png = f"{file_prefix}_differences.png"
    fig2.savefig(figurefolder / plot_filename2_pdf, dpi=300, bbox_inches="tight")
    fig2.savefig(figurefolder / plot_filename2_png, dpi=300, bbox_inches="tight")

    # Generate LaTeX snippet for subfigures
    latex_snippet = "\\begin{figure}[ht]\n"
    latex_snippet += "    \\centering\n"
    latex_snippet += "    \\begin{subfigure}[b]{0.48\\textwidth}\n"
    latex_snippet += "        \\centering\n"
    latex_snippet += f"        \\includegraphics[width=\\textwidth]{{Output/paperplots/{plot_filename1_pdf}}}\n"
    latex_snippet += "        \\caption{Feature Importance Comparison}\n"
    latex_snippet += "        \\label{fig:feature_importance_comparison}\n"
    latex_snippet += "    \\end{subfigure}\n"
    latex_snippet += "    \\hfill\n"
    latex_snippet += "    \\begin{subfigure}[b]{0.48\\textwidth}\n"
    latex_snippet += "        \\centering\n"
    latex_snippet += f"        \\includegraphics[width=\\textwidth]{{Output/paperplots/{plot_filename2_pdf}}}\n"
    latex_snippet += (
        "        \\caption{Feature Importance Differences (RaFFLE - sklearn RF)}\n"
    )
    latex_snippet += "        \\label{fig:feature_importance_differences}\n"
    latex_snippet += "    \\end{subfigure}\n"
    latex_snippet += (
        f"    \\caption{{Feature importance analysis on {dataset_name} dataset. "
    )
    latex_snippet += f"(a) Comparison of feature importances between sklearn RF (R²={r2_sklearn:.4f}$\\pm${r2_sklearn_std:.4f}) "
    latex_snippet += f"and RaFFLE (R²={r2_raffle:.4f}$\\pm${r2_raffle_std:.4f}). "
    latex_snippet += "(b) Differences in feature importance (RaFFLE - sklearn RF), sorted by absolute difference.}\n"
    latex_snippet += "    \\label{fig:feature_importance}\n"
    latex_snippet += "\\end{figure}\n"

    # Save LaTeX snippet to file
    latex_filename = f"{file_prefix}_latex.tex"
    with open(figurefolder / latex_filename, "w") as f:
        f.write(latex_snippet)

    print(f"\nFeature importance plots saved to:")
    print(f"  - {figurefolder / plot_filename1_pdf}")
    print(f"  - {figurefolder / plot_filename1_png}")
    print(f"  - {figurefolder / plot_filename2_pdf}")
    print(f"  - {figurefolder / plot_filename2_png}")
    print(f"  - {figurefolder / latex_filename} (LaTeX snippet)")

    # Print summary statistics
    print("\\n" + "=" * 80)
    print(f"Feature Importance Summary ({dataset_name})")
    print("=" * 80)
    print(f"sklearn RF Test R²:  {r2_sklearn:.4f} ± {r2_sklearn_std:.4f}")
    print(f"RaFFLE Test R²:      {r2_raffle:.4f} ± {r2_raffle_std:.4f}")
    print(
        f"\\nFeature importances correlation: {np.corrcoef(importance_sklearn, importance_raffle)[0,1]:.4f}"
    )
    print("\\nTop 3 features by sklearn RF:")
    for i, idx in enumerate(plot_order[:3], 1):
        print(f"  {i}. {feature_names[idx]}: {importance_sklearn[idx]:.4f}")
    print("\\nTop 3 features by RaFFLE:")
    raffle_order = np.argsort(importance_raffle)[::-1]
    for i, idx in enumerate(raffle_order[:3], 1):
        print(f"  {i}. {feature_names[idx]}: {importance_raffle[idx]:.4f}")
    print("\\nLargest differences (RaFFLE - sklearn RF):")
    print(
        importance_df[["Feature", "sklearn RF", "RaFFLE", "Difference"]]
        .head(5)
        .to_string(index=False)
    )


def plot_graduate_admission_importance():
    """Plot feature importance comparison for Graduate Admission dataset."""
    import urllib.request

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

    compare_feature_importance(
        X, y, feature_names, "Graduate Admission", "feature_importance"
    )


def plot_california_housing_importance():
    """Plot feature importance comparison for California Housing dataset."""
    from sklearn.datasets import fetch_california_housing

    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    compare_feature_importance(
        X, y, feature_names, "California Housing", "california_housing_importance"
    )


@click.command()
@click.option(
    "--overall_boxplot",
    is_flag=True,
    help="Plot overall boxplot with relative R2 values",
)
@click.option(
    "--typed_boxplot",
    is_flag=True,
    help="Plot linear vs non-linear boxplot with relative R2 values",
)
@click.option(
    "--transform_boxplot",
    is_flag=True,
    help="Plot R2 delta's due to power transform on numerical variables",
)
@click.option(
    "--linear_convergence",
    is_flag=True,
    help="Plot R2 on test set against number of training samples for different noise levels.",
)
@click.option(
    "--linear_convergence_high_noise",
    is_flag=True,
    help="Plot R2 on test set against number of training samples for high noise levels.",
)
@click.option(
    "--fit_duration_boxplot",
    is_flag=True,
    help="Plot boxplot with absolute fit_duration values per model",
)
@click.option(
    "--fit_duration_boxplot_log",
    is_flag=True,
    help="Plot boxplot with absolute fit_duration values per model (log scale)",
)
@click.option(
    "--relative_fit_duration_boxplot",
    is_flag=True,
    help="Plot boxplot with relative fit_duration values per model",
)
@click.option(
    "--relative_fit_duration_boxplot_log",
    is_flag=True,
    help="Plot boxplot with relative fit_duration values per model (log scale)",
)
@click.option(
    "--fit_duration_table",
    is_flag=True,
    help="Create LaTeX table with absolute fit_duration values per model per dataset",
)
@click.option(
    "--feature_importance",
    is_flag=True,
    help="Compare feature importances between RaFFLE and sklearn RF on Graduate Admission dataset",
)
@click.option(
    "--california_housing_importance",
    is_flag=True,
    help="Compare feature importances between RaFFLE and sklearn RF on California Housing dataset",
)
@click.option(
    "--relative_r2_table",
    is_flag=True,
    help="Create LaTeX table with relative R² values (with clipping)",
)
@click.option(
    "--relative_r2_table_unclipped",
    is_flag=True,
    help="Create LaTeX table with relative R² values without clipping (differences highlighted in red)",
)
@click.option(
    "--r2_pairplot",
    is_flag=True,
    help="Create pairplot of raw R² values (with clipping for CART, Lasso, Ridge)",
)
@click.option("--all", is_flag=True, help="Create all plots")
@click.pass_context
def main(
    ctx,
    overall_boxplot,
    typed_boxplot,
    transform_boxplot,
    linear_convergence,
    linear_convergence_high_noise,
    fit_duration_boxplot,
    fit_duration_boxplot_log,
    relative_fit_duration_boxplot,
    relative_fit_duration_boxplot_log,
    fit_duration_table,
    feature_importance,
    california_housing_importance,
    relative_r2_table,
    relative_r2_table_unclipped,
    r2_pairplot,
    all,
):
    if overall_boxplot or all:
        plot_overall_boxplot(ctx.obj["reltable"])
    if typed_boxplot or all:
        plot_lin_vs_nonlin_boxplot(ctx.obj["reltable"])
    if transform_boxplot or all:
        transformtable = get_transformation_table()
        plot_delta_transform(transformtable)
    if linear_convergence or all:
        plot_linear_convergence()
    if linear_convergence_high_noise or all:
        plot_linear_convergence_high_noise()
    if fit_duration_boxplot or all:
        plot_fit_duration_boxplot(ctx.obj["fit_duration_table"])
    if fit_duration_boxplot_log or all:
        plot_fit_duration_boxplot_log(ctx.obj["fit_duration_table"])
    if relative_fit_duration_boxplot or all:
        plot_relative_fit_duration_boxplot(ctx.obj["rel_fit_duration_table"])
    if relative_fit_duration_boxplot_log or all:
        plot_relative_fit_duration_boxplot_log(ctx.obj["rel_fit_duration_table"])
    if fit_duration_table or all:
        create_fit_duration_latex_table(ctx.obj["fit_duration_table"])
    if relative_r2_table or all:
        create_relative_r2_latex_table(ctx.obj["basetable"])
    if relative_r2_table_unclipped or all:
        create_relative_r2_latex_table_unclipped(ctx.obj["basetable"])
    if r2_pairplot or all:
        create_r2_pairplot(ctx.obj["basetable"])
    if feature_importance:
        plot_graduate_admission_importance()
    if california_housing_importance:
        plot_california_housing_importance()


if __name__ == "__main__":
    figurefolder.mkdir(exist_ok=True)
    basetable = load_basetable()
    reltable = get_relative_table(basetable)
    fit_duration_table = load_fit_duration_table()
    rel_fit_duration_table = get_relative_fit_duration_table(fit_duration_table)
    context = {
        "basetable": basetable,
        "reltable": reltable,
        "fit_duration_table": fit_duration_table,
        "rel_fit_duration_table": rel_fit_duration_table,
    }
    main(obj=context)
