# {{title|Timeseries (dual axis)}}  ← your template helper can support default later if you want
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib widget  # uncomment if you prefer ipympl

# Dummy data shape (user replaces with their own DataFrame)
df = pd.DataFrame({
    "ts": pd.date_range("2025-02-01", periods=500, freq="15T"),
    "a":  pd.Series(range(500)).rolling(5).mean(),
    "b":  pd.Series(range(500))[::-1].rolling(11).mean(),
}).set_index("ts")

# Params (can be overridden with %tat_example ... window=7D r1=1H r2=15T norm=zscore)
window = "{{window}}" or "7D"
r1 = "{{resample_a}}" or "1H"
r2 = "{{resample_b}}" or "30T"
norm = "{{norm}}" or ""  # "", "zscore", "minmax"

# --- smoothing/normalization using your package helpers ---
