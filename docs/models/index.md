# Supported models

InferStudio wires up two model families. They differ in more than provenance:
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

## MILES-CREDIT models

[MILES-CREDIT](https://github.com/NCAR/miles-credit) is NCAR's own AI
weather-modelling framework.

**WXFormer** is the model exposed through InferStudio. It is driven by a separate
runner and writes **ERA5-style** output: variables keep their bare names and the
vertical coordinate is a real dimension.

## Runner architecture

Both families share an abstract base:

```text
ModelRunner  (abstract)
├── Earth2StudioRunner   — dispatches into per-model uv venvs via MODEL_ENV_MAP
└── MilesCreditRunner    — dispatches into the CREDIT environment
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
