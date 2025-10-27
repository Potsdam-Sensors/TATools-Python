import importlib.resources as ir
from typing import Dict

# Minimal in-package catalog (name -> metadata)
_CATALOG: Dict[str, dict] = {
    "timeseries_dual_axis": {"path": "examples/snippets/timeseries_dual_axis.py",
                             "desc": "Dual y-axes, resample+rolling, optional normalization"},
    "multi_yaxis": {"path": "examples/snippets/multi_yaxis.py",
                       "desc": "Single X-axis with multiple Y-axes.\n\t* Call like: `%tat_example multi_yaxis n_yaxes=3 colors=[\"red\",\"blue\",\"green\"] figsize=(13,4)`"},
}

def list_examples() -> Dict[str, dict]:
    return _CATALOG.copy()

def load_example_text(name: str) -> str:
    if name not in _CATALOG:
        raise ValueError(f"Unknown example '{name}'. Try %tat_examples.")
    rel = _CATALOG[name]["path"]
    # Read from package data
    return ir.files("TATools").joinpath(rel).read_text(encoding="utf-8")
