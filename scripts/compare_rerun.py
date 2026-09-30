"""Compare a (possibly partial) rerun of the RaFFLE configurations with the original benchmark.

The benchmark writes its results file after every completed dataset, so this script can be run
while the rerun is still going. Only datasets that are present in the rerun are compared.

Example (from scripts/):
    uv run python compare_rerun.py \
        --original Output/cpilot_forest_benchmark_v11/results.csv \
        --rerun Output/cpf_v11_con_full_search/results.csv \
        --others Output/cpilot_forest_benchmark_v11_linear_models3/results.csv \
        --others Output/cart_pilot_hp_tuning/results.csv
"""

import click
import numpy as np
import pandas as pd


def _cpf(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"].str.startswith("CPF")]


def _fold_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby(["id", "model"])[col].mean().reset_index()


def _summary(x: pd.Series) -> str:
    return (
        f"mean {x.mean():+.4f} | median {x.median():+.4f} | "
        f"min {x.min():+.4f} | max {x.max():+.4f}"
    )


@click.command()
@click.option("--original", required=True, type=click.Path(exists=True), help="Original results.csv")
@click.option("--rerun", required=True, type=click.Path(exists=True), help="Rerun results.csv")
@click.option(
    "--others",
    multiple=True,
    type=click.Path(exists=True),
    help="Extra results files with competing models (for relative R2). Can be repeated.",
)
@click.option("--threshold", default=0.01, show_default=True, help="R2 change counted as notable")
@click.option("--top", default=10, show_default=True, help="Number of largest changes to list")
def main(original, rerun, others, threshold, top):
    orig = pd.read_csv(original)
    new = _cpf(pd.read_csv(rerun))
    new = new[new["max_features"] < 1]
    orig_cpf = _cpf(orig)
    replaced = orig_cpf["max_features"] < 1

    done = sorted(new["id"].unique())
    todo = sorted(set(orig_cpf.loc[replaced, "id"]) - set(done))
    print(f"Datasets finished in rerun: {len(done)} / {len(done) + len(todo)}")
    if not done:
        return

    # ------------------------------------------------------------------ per configuration
    old_r2 = _fold_mean(orig_cpf[replaced & orig_cpf["id"].isin(done)], "r2")
    new_r2 = _fold_mean(new, "r2")
    per_cfg = old_r2.merge(new_r2, on=["id", "model"], suffixes=("_old", "_new"))
    per_cfg["delta"] = per_cfg["r2_new"] - per_cfg["r2_old"]
    print("\nChange in fold-averaged R2 per rerun configuration (new - old):")
    for model, g in per_cfg.groupby("model"):
        cfg = model.replace("CPF - ", "").replace(" - n_estimators = 100", "")
        n_notable = (g["delta"].abs() > threshold).sum()
        print(f"  {cfg:70s} {_summary(g['delta'])} | |delta|>{threshold}: {n_notable}/{len(g)}")

    # ------------------------------------------------------------------ tuned RaFFLE
    kept = orig_cpf[~replaced & orig_cpf["id"].isin(done)]
    all_old = _fold_mean(orig_cpf[orig_cpf["id"].isin(done)], "r2")
    all_new = _fold_mean(pd.concat([kept, new]), "r2")

    def best(df):
        idx = df.groupby("id")["r2"].idxmax()
        return df.loc[idx].set_index("id")

    b_old, b_new = best(all_old), best(all_new)
    tuned = pd.DataFrame(
        {
            "r2_old": b_old["r2"],
            "r2_new": b_new["r2"],
            "best_old": b_old["model"].str.contains("max_node_features = 0.7").map({True: "0.7", False: "1.0"}),
            "best_new": b_new["model"].str.contains("max_node_features = 0.7").map({True: "0.7", False: "1.0"}),
        }
    )
    tuned["delta"] = tuned["r2_new"] - tuned["r2_old"]
    print("\nTuned RaFFLE (best of all CPF configurations, fold-averaged R2):")
    print(f"  delta: {_summary(tuned['delta'])}")
    print(f"  datasets with |delta| > {threshold}: {(tuned['delta'].abs() > threshold).sum()} / {len(tuned)}")
    print(
        "  best max_node_features old -> new: "
        + ", ".join(f"{o} -> {n}: {v}" for (o, n), v in tuned.groupby(["best_old", "best_new"]).size().items())
    )

    # ------------------------------------------------------------------ relative R2 (Table 2)
    extra = [pd.read_csv(f) for f in others]
    comp_orig = orig[~orig["model"].str.startswith("CPF")]
    # as in paperplots.py: tuned CART/PILOT results (cart_pilot_hp_tuning) replace the old ones
    if any(e["model"].str.startswith(("CART", "CPILOT")).any() for e in extra):
        comp_orig = comp_orig[~comp_orig["model"].str.startswith(("CART", "CPILOT"))]
    comp = pd.concat([comp_orig] + extra)
    comp = comp[comp["id"].isin(done)]
    if len(comp):
        comp_best = (
            _fold_mean(comp, "r2")
            .assign(r2=lambda d: d["r2"].clip(0, 1))
            .groupby("id")["r2"]
            .max()
        )
        rel = pd.DataFrame(
            {
                "old": tuned["r2_old"].clip(0, 1) / np.maximum(tuned["r2_old"].clip(0, 1), comp_best),
                "new": tuned["r2_new"].clip(0, 1) / np.maximum(tuned["r2_new"].clip(0, 1), comp_best),
            }
        ).dropna()
        print(
            f"\nRelative R2 of tuned RaFFLE on these {len(rel)} datasets "
            "(w.r.t. the best of RaFFLE and the models in --original/--others):"
        )
        print(f"  old: mean {rel['old'].mean():.3f}, sd {rel['old'].std():.3f}, min {rel['old'].min():.3f}")
        print(f"  new: mean {rel['new'].mean():.3f}, sd {rel['new'].std():.3f}, min {rel['new'].min():.3f}")
        print(f"  RaFFLE best method on: old {(rel['old'] >= 1).sum()}, new {(rel['new'] >= 1).sum()} datasets")
        if not others:
            print("  (no --others given: linear models and the tuned CART/PILOT runs are not included)")

    # ------------------------------------------------------------------ fit duration
    old_t = _fold_mean(orig_cpf[replaced & orig_cpf["id"].isin(done)], "fit_duration")
    new_t = _fold_mean(new, "fit_duration")
    t = old_t.merge(new_t, on=["id", "model"], suffixes=("_old", "_new"))
    ratio = t["fit_duration_new"] / t["fit_duration_old"]
    print(
        f"\nFit duration ratio new/old for the rerun configurations: "
        f"median {ratio.median():.2f}, 90% quantile {ratio.quantile(0.9):.2f}, max {ratio.max():.2f}"
    )
    print("  (only meaningful if the rerun runs on the same hardware as the original)")

    # ------------------------------------------------------------------ largest changes
    print(f"\nLargest changes in tuned RaFFLE R2 (top {top}):")
    names = orig.drop_duplicates("id").set_index("id")["name"] if "name" in orig else None
    show = tuned.reindex(tuned["delta"].abs().sort_values(ascending=False).index).head(top)
    if names is not None:
        show = show.join(names)
    print(show.round(4).to_string())


if __name__ == "__main__":
    main()
