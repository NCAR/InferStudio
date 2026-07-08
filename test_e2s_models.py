#!/usr/bin/env python3
"""
Test earth2studio model installations.

Checks each model in EARTH2STUDIO_MODELS for:
  1. Import success
  2. from_pretrained() load (downloads weights on first run)
  3. 1-step inference with random data (no network data source needed)

Run with the target venv:
    /glade/work/pearse/E2S/.venv/bin/python test_e2s_models.py

Optional: test only specific models:
    /glade/work/pearse/E2S/.venv/bin/python test_e2s_models.py AIFS Pangu

Weights cache location is controlled by the EARTH2STUDIO_CACHE env var
(defaults to ~/.cache/earth2studio). Point it at a shared GLADE path to
avoid re-downloading per-user:
    export EARTH2STUDIO_CACHE=/glade/work/pearse/E2S/.cache
"""

import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Model registry
# Maps the label used in EARTH2STUDIO_MODELS to the earth2studio class and
# any special kwargs needed for from_pretrained().
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "AIFS":          {"cls_path": "earth2studio.models.px.AIFS"},
    "Aurora":        {"cls_path": "earth2studio.models.px.Aurora"},
    "FourCastNet3":  {"cls_path": "earth2studio.models.px.FCN3"},
    "GraphCast":     {"cls_path": "earth2studio.models.px.GraphCastOperational"},
    "Pangu":         {"cls_path": "earth2studio.models.px.Pangu24"},   # 24-hr variant
    "SFNO":          {"cls_path": "earth2studio.models.px.SFNO"},
}

EARTH2STUDIO_MODELS = {"AIFS", "Aurora", "FourCastNet3", "GraphCast", "Pangu", "SFNO"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def import_cls(cls_path: str):
    """Dynamically import a class from a dotted path."""
    module_path, cls_name = cls_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def make_random_data_source(model):
    """
    Build an earth2studio Random data source shaped to match the model's
    input_coords (lat/lon grid).  Falls back gracefully if input_coords
    doesn't expose lat/lon.
    """
    from earth2studio.data import Random

    try:
        ic = model.input_coords()
        domain_coords = OrderedDict()
        for key in ("lat", "lon", "latitude", "longitude"):
            if key in ic:
                domain_coords[key] = ic[key]
        if not domain_coords:
            # Generic fallback: 1-degree global grid
            domain_coords = OrderedDict(
                lat=np.linspace(90, -90, 181),
                lon=np.linspace(0, 359, 360),
            )
    except Exception:
        domain_coords = OrderedDict(
            lat=np.linspace(90, -90, 181),
            lon=np.linspace(0, 359, 360),
        )

    return Random(domain_coords)


def run_one_step(model, device):
    """Run a single inference step with random data. Returns elapsed seconds."""
    from earth2studio.data import Random
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic

    data = make_random_data_source(model)

    # Use an in-memory zarr store so no disk I/O is needed
    io = ZarrBackend()

    t0 = time.perf_counter()
    deterministic(
        time=["2024-01-01T00:00:00"],
        nsteps=1,
        prognostic=model,
        data=data,
        io=io,
        device=device,
        #verbose=False,
    )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-model test
# ---------------------------------------------------------------------------

def test_model(name: str, device: torch.device) -> dict:
    result = {"name": name, "import": False, "load": False, "infer": False,
              "error": None, "elapsed": None}

    cfg = MODEL_REGISTRY.get(name)
    if cfg is None:
        result["error"] = f"No entry in MODEL_REGISTRY for '{name}'"
        return result

    # 1. Import
    try:
        cls = import_cls(cfg["cls_path"])
        result["import"] = True
    except Exception as e:
        result["error"] = f"Import failed: {e}"
        return result

    # 2. Load weights via from_pretrained()
    print(f"  [{name}] Loading weights (may download on first run)…")
    try:
        model = cls.from_pretrained()
        model.eval()
        result["load"] = True
    except Exception as e:
        result["error"] = f"from_pretrained() failed: {traceback.format_exc()}"
        return result

    # 3. One-step inference
    print(f"  [{name}] Running 1-step inference on {device}…")
    try:
        elapsed = run_one_step(model, device)
        result["infer"] = True
        result["elapsed"] = elapsed
    except Exception as e:
        result["error"] = f"Inference failed: {traceback.format_exc()}"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models_to_test = sys.argv[1:] if len(sys.argv) > 1 else sorted(EARTH2STUDIO_MODELS)
    unknown = set(models_to_test) - set(MODEL_REGISTRY)
    if unknown:
        print(f"WARNING: unknown model names (will be skipped): {unknown}")
        models_to_test = [m for m in models_to_test if m in MODEL_REGISTRY]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nearth2studio model test")
    print(f"  device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  models : {models_to_test}\n")
    print("─" * 60)

    results = []
    for name in models_to_test:
        print(f"\n▶ {name}")
        r = test_model(name, device)
        results.append(r)

    # Summary table
    print("\n" + "─" * 60)
    print(f"{'Model':<16} {'Import':>6} {'Load':>6} {'Infer':>6} {'Time':>8}  Notes")
    print("─" * 60)
    all_ok = True
    for r in results:
        ok   = lambda v: "✓" if v else "✗"
        t    = f"{r['elapsed']:.1f}s" if r["elapsed"] is not None else "—"
        note = ""
        if r["error"]:
            # Truncate long tracebacks in the summary
            first_line = r["error"].splitlines()[-1] if r["error"] else ""
            note = first_line[:50]
            all_ok = False
        print(f"{r['name']:<16} {ok(r['import']):>6} {ok(r['load']):>6} "
              f"{ok(r['infer']):>6} {t:>8}  {note}")

    print("─" * 60)
    if all_ok:
        print("\n✓ All models passed.")
    else:
        print("\n✗ Some models failed — see errors above.")
        for r in results:
            if r["error"]:
                print(f"\n{'='*60}\n[{r['name']}] ERROR DETAIL:\n{r['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
