# Default degrees of freedom settings for PILOT
# These control the complexity of different node types in the tree
DEFAULT_DF_SETTINGS = {
    "con": 1,   # Constant node (no split)
    "lin": 2,   # Linear node (simple linear model)
    "pcon": 5,  # Piecewise constant
    "blin": 5,  # Bilinear
    "plin": 7,  # Piecewise linear
    "pconc": 5, # Piecewise constant constrained
}
