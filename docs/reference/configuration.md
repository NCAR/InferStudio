# Configuration reference

## Module layout

```text
InferStudio/
├── app_layout.py            # entry point; sidebar + tab assembly, dataset scanning
├── inferenceTab.py          # run configuration, per-model log tabs, status row
├── datasetSelector2.py      # dataset browser with single-select + checkbox sync
├── dataset_metadata.py      # metadata extraction and convention detection
├── datasetPlot.py           # generic dataset plotting
├── era5_plot.py             # ERA5-style (CREDIT) plot rendering
├── earth2StudioPlot.py      # Earth2Studio flattened-format plot rendering
├── earth2StudioRunner.py    # Earth2Studio dispatch; MODEL_ENV_MAP
├── milesCreditRunner.py     # MILES-CREDIT dispatch; ModelRunner base
└── static/
    └── styles.css           # layout overrides
```

:::{important}
`earth2StudioPlot.py` and `earth2StudioRunner.py` use a **capital S**. Imports and
filenames must match exactly — GLADE is case-sensitive, and a lowercase `s` in an
import produces a `ModuleNotFoundError` that reads like a missing dependency.
:::

## Paths

| Purpose | Default |
| --- | --- |
| Application code | `/glade/work/$USER/InferStudio/` |
| Panel conda environment | `/glade/work/$USER/conda-envs/creditJun3/` |
| Per-model uv venvs | `/glade/work/$USER/E2S/envs/<model>/` |
| Default output directory | `/glade/scratch/$USER` |
| Debug log | `/tmp/debug.log` |

Output for a run lands in `<output-directory>/<simulation-name>/`.

:::{warning}
GLADE scratch is subject to a rolling purge. Anything you want to keep must be
moved to work or campaign storage.
:::

## Serving options

```bash
panel serve app_layout.py \
    --port 5006 \
    --allow-websocket-origin='<host>:<port>' \
    --num-procs 1
```

| Flag | Why |
| --- | --- |
| `--port` | Must match the port you forward or proxy. |
| `--allow-websocket-origin` | Required behind a proxy or tunnel; Panel rejects mismatched origins. |
| `--num-procs 1` | Keep it at one. Multiple processes do not share the in-memory dataset cache or run state, so a user's session can land on a worker that knows nothing about their run. |
| `--autoreload` | Development only. It restarts on file change, which kills in-flight inference. |
| `--static-dirs static=./static` | Needed if the stylesheet is not being picked up. |

## Styling

Layout overrides live in `static/styles.css`. Where Python-side sizing and CSS
disagree, CSS rules marked `!important` win — this is the reliable lever for
sizing problems that Panel's `sizing_mode` will not fix, particularly inside
Bokeh `Tabs`, whose shadow DOM does not propagate `stretch_both`.

## Debug logging

The app writes to `/tmp/debug.log`, truncated at the start of each run so the file
only ever contains the current run:

```python
open("/tmp/debug.log", "w").close()
```

Tail it in a second terminal while reproducing a problem:

```bash
tail -f /tmp/debug.log
```

Because it is truncated per run, capture a copy before starting another run if you
are filing a bug.

## Environment variables

| Variable | Effect |
| --- | --- |
| `USER` | Used to construct the default output path. |
| `CUDA_VISIBLE_DEVICES` | Restricts which GPUs child processes see. Useful for running two models on distinct GPUs of a multi-GPU node. |
| `PBS_JOBID` | Present when running inside a batch or interactive allocation; useful for confirming which job the server belongs to. |
