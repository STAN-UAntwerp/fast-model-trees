# API reference

```{eval-rst}
.. currentmodule:: pilot
```

## PILOT

```{eval-rst}
.. autoclass:: PILOT
   :members: train, predict, tree_summary, feature_importances_
```

## RaFFLE

```{eval-rst}
.. autoclass:: RaFFLE
   :members: fit, predict, oob_predict, oob_mask, estimators_samples_, feature_importances_
```

## PILOTWrapper

```{eval-rst}
.. autoclass:: PILOTWrapper
   :members: predict, tree_summary, feature_importances_
```

## Constants

```{eval-rst}
.. data:: DEFAULT_DF_SETTINGS
   :type: dict[str, int]
   :value: {"con": 1, "lin": 2, "pcon": 5, "blin": 5, "plin": 7, "pconc": 5}

   Default degrees of freedom of the node types, used by :class:`PILOT` when
   ``df_settings`` is not given and as the base that ``alpha`` scales in
   :class:`RaFFLE`. The order of the keys is the order of ``df_settings``.
```

## Aliases

`RandomForestCPilot` and `CPILOTWrapper` are kept as aliases of {class}`RaFFLE` and
{class}`PILOTWrapper` for backward compatibility.
