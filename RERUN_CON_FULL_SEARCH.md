# Reruns after the change to the con rule (branch `con-extended-search`)

## What changed

In a random PILOT tree, a node in which the constant model (con) has the lowest BIC among the
models on the random subset of `n_features_node` features is no longer turned into a leaf right
away. The model search is first extended to all features of the tree. The node only becomes a
con leaf if con remains optimal; otherwise the best non-constant model over all features is
fitted and the recursion continues. This is the variant covered by Theorem 1 of the RaFFLE paper.

* Controlled by `con_full_search` (default `True`) in `RaFFLE`, `PILOTWrapper`, `PILOT` and the raw
  C++ class. `con_full_search=False` reproduces version 0.1.1 exactly.
* The extended search is only triggered when `n_features_node < n_features_tree`. With all features
  per node the fitted trees are bit-for-bit identical to 0.1.1 (the random number stream is unchanged).
* `RaFFLE.con_search_stats_` reports how often con was optimal on the subset and how often the
  extended search overruled it.
* Also fixed: when a lin fit is discarded because of `rel_tolerance`, its RSS reduction is no longer
  kept in the feature importance.

## Which paper results are affected

| Result in the paper | Affected? | Why |
|---|---|---|
| Table 2, Fig. 4 (boxplots overall), Fig. 6 (linear vs nonlinear), appendix tables of relative R² | **Yes** | Tuned RaFFLE takes the best of a grid that includes `max_node_features = 0.7` (best config in 54 of 137 datasets) |
| Fig. 7 (fit duration) | **Yes** | RaFFLE's time is taken from its fastest configuration, which can be a 0.7 configuration |
| dRaFFLE (all figures/tables) | No | Uses `max_node_features = 1.0` |
| CART, PILOT, RF, XGB, Lasso, Ridge | No | Not RaFFLE |
| Fig. 5 (pairs plot CART vs Lasso/Ridge) | No | No RaFFLE |
| Fig. 3 and appendix figure (linear convergence) | No | `n_features_node = 1.0` |
| Fig. 8 (California Housing importances), R² 0.73 | No | `n_features_node = 1.0`; rerun with the new code gives the same importances and R² to 4 decimals |

The power-transform benchmark (`benchmark_power_transform_v2`) also contains 0.7 configurations,
but it is not used in the paper. Rerunning it is optional.

## Commands

Build the extension from this branch first, then from `scripts/`:

```bash
# 1. Main benchmark: only the RaFFLE configurations with max_node_features = 0.7,
#    without power transform (as in cpilot_forest_benchmark_v11).
#    6 configurations x 5 folds x 137 datasets; the original 0.7 runs took about 24 h of fit time.
uv run python benchmark.py -e cpf_v11_con_full_search -m cpf -nf 0.7 --no_power_transform

# 2. (optional, not in the paper) power-transform benchmark
uv run python benchmark.py -e cpf_power_transform_v2_con_full_search -m cpf -nf 0.7 --power_transform

# 3. Regenerate the figures and tables
uv run python paperplots.py --all
```

`paperplots.py` reads the original result files as before and, if the rerun folders above exist,
replaces the CPF rows with `max_node_features < 1` by the rerun results (it prints a warning if the
rerun does not cover exactly the same dataset/fold/model cells). The original result files are in the
git history (`git show 64cfe6c^:Output/cpilot_forest_benchmark_v11/results.csv`) if they are not
present locally.

Run the rerun on the same machine as the original benchmark if possible, otherwise the fit durations
in Fig. 7 mix hardware. If that is not possible, rerun all models for the timing figure.

Then update in the manuscript: the numbers in Table 2 and the text of Section 4.2 that quotes them
(RaFFLE mean 0.99, sd 0.02; "always above 80%"), Figs. 4, 6 and 7, and the appendix tables.
Figure numbers refer to `RaFFLE_final.tex`.
