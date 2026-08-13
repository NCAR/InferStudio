# InferStudio

**InferStudio** is a browser-based application for running AI weather-model
inference on NSF NCAR HPC resources and comparing the results visually — without
writing an inference script, building a Python environment, or hand-editing a
PBS submission.

It is built with [HoloViz Panel](https://panel.holoviz.org/) and runs on a
Casper compute node, reached through [NCAR's Open OnDemand portal](https://ondemand.hpc.ucar.edu/).
From InferStudio's Inference tab you can pick a model, an initialization time, and a forecast
length; InferStudio dispatches the run into the correct pre-built environment,
streams the log back into the browser, and then loads the output for plotting
and scoring against a reference analysis.

```{card} New here?
:link: getting-started/quickstart
:link-type: doc

Start with the **Quickstart** — launch the app on Casper and run your first
deterministic forecast in about ten minutes.
```

## What problem it solves

Running a modern AI weather model normally means solving four unrelated
problems before you see a single field:

1. **Environment resolution.** Each model in the
   [Earth2Studio](https://nvidia.modulus.github.io/earth2studio/) ecosystem
   pins a mutually incompatible dependency stack.  For example ONNX Runtime for Pangu,
   `makani` plus `torch-harmonics` compiled from source for SFNO and
   FourCastNet3, `flash_attn` for AIFS to name a few. On a shared HPC system these stacks
   also have to respect the site CUDA driver ceiling.
2. **Driver code.** Calling `earth2studio.run.deterministic()` correctly
   requires knowing which data source to hand it, how to apply an
   initialization time to an available analysis cycle, and how to configure the
   IO backend.
3. **Job placement.** GPU inference needs the right queue, the right account
   code, and a session that survives an SSH disconnect.
4. **Output wrangling.** Different model families write structurally different
   NetCDF: Earth2Studio flattens level and variable into one name (`u850`) and
   carries a `lead_time` dimension, while MILES-CREDIT models write ERA5-style
   files with a separate vertical coordinate.

InferStudio absorbs all four. The user interface runs custom inference, stores data
on glade, and presents a set of plots.

## Who it is for

:::{list-table}
:header-rows: 1
:widths: 22 78

* - Persona
  - How InferStudio is used
* - **Research scientist**
  - Screen several AI models against each other and against a NWP reference for
    a case of interest; export RMSE, CRPS, and spread/error ratio for a
    specific variable and lead time.
* - **Instructor**
  - Run a live forecast in front of a class and step through lead times without
    the lecture becoming a debugging session.
* - **Student / new user**
  - Get a first forecast out of a model on day one, then read the generated
    command to learn what the underlying API call looks like.
* - **Model developer**
  - Sanity-check a new checkpoint or a candidate `makani` branch against the
    established models using an identical initialization and scoring path.
:::

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting-started/prerequisites
getting-started/launching
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

user-guide/interface
user-guide/running-inference
user-guide/datasets
user-guide/visualization
user-guide/statistics
```

```{toctree}
:maxdepth: 2
:caption: Models
:hidden:

models/index
models/environments
models/adding-a-model
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/output-formats
reference/configuration
reference/troubleshooting
reference/api
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

contributing
changelog
```

## Index

- {doc}`user-guide/interface` — a tour of every control in the app
- {doc}`models/index` — which models are wired up and what each one expects
- {doc}`reference/output-formats` — the two NetCDF conventions you will meet
- {doc}`reference/troubleshooting` — queue limits, tunnels, and stuck runs
