<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo/wordmark_dark_full.png">
    <img alt="InferStudio" src="static/logo/wordmark_light_full.png" width="480">
  </picture>
</p>

# InferStudio

InferStudio is NSF NCAR's web application for running AI weather model
inference and visualizing the results. It's run with
[Panel](https://panel.holoviz.org/) and runs on NCAR's Casper HPC system via
[Open OnDemand](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/ood/sandbox-apps-and-appverse/#overview), giving researchers a single interface to launch inference for
multiple AI weather models, browse the resulting forecasts, and verify them
against GFS analysis — without needing to manage each model's environment or
write plotting code by hand.

## Supported models

| Model | Backend | Status |
|---|---|---|
| AIFS | earth2studio | Fully supported (visualization + verification) |
| Aurora | earth2studio | Fully supported (visualization + verification) |
| Pangu | earth2studio | Fully supported (visualization + verification) |
| WXFormer | MILES-CREDIT | Inference supported; visualization not yet wired up (different output format) |

## The two tabs

### Visualization

- **Datasets** — browse available simulation runs (each one a folder
  containing one subfolder per model, e.g. `AIFS/`, `Aurora/`). Only one
  dataset can be viewed at a time.
- **Variable & Level** — a single shared set of controls (Time, Variable,
  Level, Colormap) drives every plot on screen at once, so comparing models
  means looking at the exact same field, level, and forecast step across all
  of them.
- **Field plots** — one panel per model, each labeled with the actual valid
  time and forecast lead time (not just a raw time-step index).
- **Forecast Verification Statistics** — on request, computes RMSE and CRPS
  (which, for a single deterministic forecast, is equivalent to MAE) for the
  current variable/level at every available lead time, verified against GFS
  analysis fetched via earth2studio. Handles grid mismatches between models
  (e.g. some models use a 720-point latitude grid, GFS uses 721) via
  interpolation.
- **Metadata** — quick reference for a selected dataset: time range, grid
  size, available variables, number of forecast steps, etc.

An example dataset ships with the app and is selected automatically on
startup, so the Visualization tab has something to show before you've run
any inference yourself.

### Inference

- **AI Model** — choose one or more models to run.
- **Output Parameters** — simulation name (auto-generated from the selected
  models and a timestamp, but editable), output directory, and which
  surface/upper-air variables to save. 
- **Time Settings** — pick a start date and a lead time (in forecast steps);
  combined with the time-step increment, this determines the run's end date
  automatically.
- **Launcher** — runs each selected model sequentially, showing live status
  (pending/running/done/error/cancelled) and streaming console output per
  model. Runs can be cancelled mid-flight. When a run finishes, it's
  automatically selected back in the Visualization tab.

## Project layout

| File | Purpose |
|---|---|
| `app_layout.py` | Top-level page layout, dataset scanning, and wiring between tabs |
| `datasetPlot.py` | Shared plot controls and per-dataset field plot rendering |
| `earth2StudioPlot.py` | Reads earth2studio-format NetCDF output and renders individual field plots |
| `forecastStatsPanel.py` | RMSE/CRPS verification against GFS analysis |
| `inferenceTab.py` | Inference tab logic: model runs, status tracking, console output |
| `outputParams.py` | Simulation name, output directory browser, variable selection |
| `timePicker.py` | Start date / lead time / time-step-increment controls |
| `datasetSelector2.py` | Dataset browser (checkbox list with single-selection) |
| `metadata.py` | Metadata display panel |
| `earth2StudioRunner.py`, `milesCreditRunner.py` | Per-backend model execution |
| `static/styles.css` | App-wide styling |

## Running InferStudio

From a Jupyter notebook cell:

```python
from pathlib import Path
import panel as pn
from app_layout import build_app

DATA_DIR = Path("/path/to/your/output/directory")
pn.extension('modal', notifications=True)

template = build_app(DATA_DIR)
template.servable()
```

Each AI model backend (AIFS, Aurora, Pangu, WXFormer) runs from its own
per-model Python environment on Casper, since the models have incompatible
dependency stacks (different CUDA/PyTorch/library requirements).

## Known limitations

- WXFormer output uses MILES-CREDIT's ERA5-style format (a true `level`
  dimension rather than earth2studio's flattened `u850`-style variables) and
  isn't yet plotted or verified — it currently shows a placeholder in the
  Visualization tab.
- Only one dataset can be viewed at a time in the Visualization tab.
- Forecast verification requires network access to fetch GFS analysis data,
  and can take a noticeable amount of time for datasets with many forecast
  steps.
