# Supported models

InferStudio wires up two families of AI models, plus physical (NWP) models
fetched for comparison. They differ in more than provenance:
they use different Python stacks, different output conventions, and different
runner classes inside the application.

## Earth2Studio models

[Earth2Studio](https://nvidia.modulus.github.io/earth2studio/) is NVIDIA's
inference framework for AI weather and climate models. InferStudio drives it
through a generated script that calls `earth2studio.run.deterministic()` (or the
ensemble equivalent) with an explicit data source.

:::{list-table}
:header-rows: 1
:widths: 16 24 60

* - Model
  - Backend
  - Notes
* - **Pangu-Weather**
  - ONNX Runtime
  - Requires a GPU-enabled ONNX Runtime build. Pinned to `onnxruntime-gpu==1.21.0`
    in its environment.
* - **AIFS**
  - PyTorch
  - ECMWF's AI forecasting system. Needs `flash_attn` compiled from source.
* - **Aurora**
  - PyTorch
  - Microsoft's foundation model.
* - **SFNO**
  - PyTorch + `makani`
  - Spherical Fourier Neural Operator. The heaviest environment to build:
    `makani` from GitHub, `warp-lang`, `nvidia-physicsnemo`, and
    `torch-harmonics` compiled from source with `--no-build-isolation --no-deps`.
* - **FourCastNet 3**
  - PyTorch + `makani`
  - Requires a specific matched pair of `makani` and `torch-harmonics` commits.
    See {doc}`environments`.
:::

All of these write **flattened** output: level and variable are combined into a
single name (`u850`, `t500`, `z500`), and the file carries a `lead_time`
dimension.

:::{note}
The model roster tracks what has actually been built and validated on Casper, not
everything Earth2Studio supports upstream. Availability of a new upstream model
is a matter of building its environment — see {doc}`adding-a-model`.
:::

## Climate models

The Inference tab's **Climate** sub-tab runs climate emulators for months to
years, 6-hourly, from 00Z on the 1st of a month. Its start-date picker only
offers months that every selected model can start from.

:::{list-table}
:header-rows: 1
:widths: 18 30 52

* - Model
  - Starts from
  - Notes
* - **CAMulator**
  - CAM states shipped with it: 1 January or 1 July, 1980-2014
  - NSF NCAR's CAM6 emulator, 1 degree, 32 model levels. Cyclic or transient
    (1980-2014) forcing.
* - **DLESyM v0** (`DLESyMv0_ISCCP_ERA5`)
  - ERA5 (ARCO), any month from February 1940 to December 2022
  - University of Washington's coupled atmosphere-ocean model on a HEALPix
    grid of about 1 degree. No 3D fields or surface pressure. Each step is 4
    days, so it runs through earth2studio's iterator rather than
    `deterministic()`; the ocean's SST is valid every 48 hours and is held at
    its last value in between. Its 0.25 degree output is interpolated from
    the HEALPix grid, so it's saved every 4th point, on a 1 degree grid.
:::

A variable the Climate tab offers that a model doesn't have is left out of its
output, with a warning in its log.

## MILES-CREDIT models

[MILES-CREDIT](https://github.com/NCAR/miles-credit) is NCAR's own AI
weather-modelling framework.

**WXFormer** is the model exposed through InferStudio. It is driven by a separate
runner and writes **ERA5-style** output: variables keep their bare names and the
vertical coordinate is a real dimension.

## Physical models

**GFS**, NOAA's operational Global Forecast System, is the physics-based
reference. It isn't run: `ForecastSourceRunner` downloads GFS's published
forecast for the same initialization, lead times and variables through
earth2studio's `GFS_FX` source, and writes it into the suite in the same layout
as the Earth2Studio models. Plots, differences, video export and statistics
then treat it as one more model.

- Forecasts are on NOAA's AWS archive from the 2021-02-27 00Z cycle on, out to
  384 h: hourly to 120 h, 3-hourly after.
- It needs no GPU. Alongside AI models it runs in a CPU lane of the same job;
  on its own it goes to a small CPU-only job on Casper.
- Verified against GFS analysis it is scored against its own starting point;
  see {doc}`../user-guide/statistics`.

Another earth2studio forecast source (`IFS_FX`, `GEFS_FX`, ...) is an entry in
`FORECAST_SOURCES` in `inference/forecastSourceRunner.py`, plus its date and
lead-time checks.

## Runner architecture

Both families share an abstract base:

```text
ModelRunner  (abstract)
├── Earth2StudioRunner   — dispatches into per-model uv venvs via MODEL_ENV_MAP
├── MilesCreditRunner    — dispatches into the CREDIT environment
└── ForecastSourceRunner — fetches a physical model's published forecast
```

A runner is responsible for:

1. Resolving the model name to an interpreter path
2. Generating the inference script for the selected initialization and lead time
3. Launching it as a subprocess in a new session
4. Streaming stdout/stderr back to the UI
5. Reporting completion status and the output path

Adding a model family means adding a `ModelRunner` subclass. Adding a model to an
existing family means adding an environment and a map entry.

## Choosing between them

There is no single best model, which is the reason InferStudio exists. Some
practical guidance for a first comparison:

- Pangu is fast and a reasonable baseline.
- SFNO and FourCastNet 3 behave differently at long leads than the transformer
  models; include at least one of each family.
- WXFormer is the relevant comparison if your question is about NCAR's own model
  development rather than about the published models.
- Whatever set you choose, run them from the same initialization to the same lead
  time, or the statistics panel will produce a table that looks valid and is not.
