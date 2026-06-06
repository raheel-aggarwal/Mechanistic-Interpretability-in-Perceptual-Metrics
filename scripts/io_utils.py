"""
io_utils.py
-----------
Serialisation and deserialisation of decomposition outputs.

Public API
----------
save_outputs(grounding, decomp_results, attributions, output_dir, cfg)
load_outputs(output_dir, cfg) -> (grounding, decomp_results, attributions)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Default filenames — can be overridden by passing a cfg dict.
_DEFAULT_FNAMES = {
    "grounding_fname":    "grounding.pkl",
    "components_fname":   "components.pkl",
    "attributions_fname": "attributions.json",
}


def save_outputs(
    grounding:      Dict,
    decomp_results: Dict,
    attributions:   Dict,
    output_dir:     Path,
    cfg:            Dict = None,
) -> None:
    """
    Persist grounding values, decomposed components, and attributions to disk.

    Files written
    -------------
    grounding.pkl      — relu gates, pool inputs, conv outputs
    components.pkl     — layer_components (numpy) + metadata
    attributions.json  — attribution summaries
    """
    cfg        = cfg or {}
    fnames     = {**_DEFAULT_FNAMES, **cfg}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grounding
    with open(output_dir / fnames["grounding_fname"], "wb") as f:
        pickle.dump(grounding, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Components: convert tensors → numpy for portability
    comp_np = {
        k: v.numpy() if isinstance(v, torch.Tensor) else v
        for k, v in decomp_results["layer_components"].items()
    }
    meta = {k: v for k, v in decomp_results.items()
            if k != "layer_components"}
    with open(output_dir / fnames["components_fname"], "wb") as f:
        pickle.dump({"components": comp_np, "meta": meta}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # Attributions — numpy → list for JSON
    def _to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_json(v) for k, v in obj.items()}
        return obj

    with open(output_dir / fnames["attributions_fname"], "w") as f:
        json.dump(_to_json(attributions), f, indent=2)

    g_path = output_dir / fnames["grounding_fname"]
    c_path = output_dir / fnames["components_fname"]
    print(f"Saved outputs to {output_dir.resolve()}")
    print(f"  {g_path.name:<30s}  {g_path.stat().st_size / 1024:8.1f} kB")
    print(f"  {c_path.name:<30s}  {c_path.stat().st_size / 1024:8.1f} kB")
    print(f"  {fnames['attributions_fname']}")


def load_outputs(
    output_dir: Path,
    cfg:        Dict = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Load previously saved decomposition outputs.

    Returns
    -------
    grounding, decomp_results, attributions
    """
    cfg        = cfg or {}
    fnames     = {**_DEFAULT_FNAMES, **cfg}
    output_dir = Path(output_dir)

    with open(output_dir / fnames["grounding_fname"], "rb") as f:
        grounding = pickle.load(f)

    with open(output_dir / fnames["components_fname"], "rb") as f:
        saved = pickle.load(f)

    layer_components = {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in saved["components"].items()
    }
    decomp_results = saved["meta"]
    decomp_results["layer_components"] = layer_components

    with open(output_dir / fnames["attributions_fname"], "r") as f:
        attributions = json.load(f)

    print(f"Loaded outputs from {output_dir.resolve()}")
    return grounding, decomp_results, attributions
