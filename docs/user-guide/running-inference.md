# Running inference

## What happens when you press Run

The InferStudio session itself runs on CPUs. Pressing **Run Inference** submits
a separate PBS batch job that runs the selected models on GPUs, charged to the
project account you chose when starting InferStudio from Open OnDemand.

The **Compute Settings** section sets up that job:

- **System** — Derecho (the default) or Casper. The session itself always runs on Casper, but
  the job can go to either; output lands on GLADE either way.
- **GPU type** — A100 40GB on Derecho; A100 80GB (Casper's default), H100 or
  L40 on Casper. Some models, such as AIFS, need more memory than a 40-45 GB card has,
  so they need one of the 80 GB types.
- **GPUs** — with more than one, the models are split between the GPUs and run
  side by side; each GPU still runs its own models one after another. There is
  no point asking for more GPUs than models, so the choice stops there.
- **Wall time** — estimated from the models and the forecast length, with a
  safety margin, rounded up to 15 minutes, and capped at the system's limit
  (24 h on Casper, 12 h on Derecho). If the cap cuts the estimate short, the
  label says so.

The job ends as soon as the last model finishes, so the GPUs aren't held for the
rest of the wall time.

InferStudio does not import the models into its own process. Doing so would be
impossible — the dependency stacks conflict — so instead the job runs each model
with a *different Python interpreter*.

```text
PBS job  (conda: the session's environment)
    │
    ├─ GPU 0:  /glade/work/$USER/E2S/envs/aifs/bin/python   <script>   then the next model on GPU 0
    ├─ GPU 1:  /glade/work/$USER/E2S/envs/pangu/bin/python  <script>
    └─ ...
             │
             └─ <output>/<model>/<model>_run.log ──▶ followed into the model's log tab
```

The mapping from model to interpreter lives in a `MODEL_ENV_MAP` dictionary in
the Earth2Studio runner module. The generated script is what actually calls
`earth2studio.run.deterministic()` (or the ensemble equivalent) with the model
package, the data source, and the IO backend.

The job's own files — the batch script, each model's command, and PBS's output —
go in `~/.inferstudio/jobs/<simulation name>_<time>/`. The first lines of each
model's log tab give the job ID and that directory.

### Queue time and elapsed time

The Launcher shows the job's ID and state, and two timers: **Time in queue**,
from submission until the job starts, and **Elapsed**, from then until it ends.
A long wait for GPUs is a busy queue, not slow inference.

**Cancel** deletes the job (`qdel`), whether it is still queued or running.

If the browser tab is closed, or the session ends, while the job is running, the
job carries on and writes its output; load it later with **Load Existing
Suite**.

## Choosing an initialization time

Earth2Studio's default data source pulls GFS/GDAS analyses, which are published
at 00, 06, 12, and 18 UTC only. The time picker enforces this by snapping any
selection to the nearest available cycle before the run is built. You will see
the resolved time reflected in the generated command in the log.

Practical limits:

- **Too recent.** The analysis for a cycle is not available the moment the cycle
  time passes. Give it a few hours.
- **Too old.** Fast-access archives are finite. Deep-archive dates may require a
  different data source.

## Choosing a lead time

The lead-time slider is in forecast hours. Cost is roughly linear in lead time:
each step is one autoregressive model call. A 10-day (240 h) forecast at 6-hourly
output is 40 steps.

:::{tip}
When you are comparing models for the first time, run a short lead (24–48 h)
first. Environment problems and data-source problems both surface within the
first step, so a short run is a cheap smoke test.
:::

## Deterministic vs ensemble

- **Deterministic** — a single trajectory. Produces RMSE-scoreable output.
- **Ensemble** — perturbed initial conditions, one trajectory per member.
  Required for CRPS and for the spread/error ratio. Cost and disk scale with
  member count.

## Reading the log

The log tab is the raw child process output. On a first run for a given model,
expect a substantial download phase as the model package is fetched and cached —
this is not a hang. Subsequent runs skip it.

Things worth recognizing in the log:

| Log content | Meaning |
| --- | --- |
| Package download progress bars | First-run model fetch; will be cached |
| `CUDA error: no kernel image is available` | Compiled extension does not match the GPU architecture — an environment problem, see {doc}`../models/environments` |
| `undefined symbol` on import of a compiled extension | Torch/extension ABI mismatch in that model's venv |
| Data source HTTP 404 | Initialization time has no published analysis |
| `torch.OutOfMemoryError` | Another process is sharing the GPU, or the member count is too high |

## Cancelling and re-running

**Cancel** deletes the PBS job, stopping every model in it. Partial output may
remain in the output directory; delete it before re-running with the same
simulation name, or pick a new name.

Re-running with an identical name overwrites. The auto-generated names include
the model and initialization time, which makes accidental collisions unlikely but
not impossible when you iterate on the same case.

## Physical models

The model box has a second row, **Physical models**, with GFS. Selecting it
doesn't run GFS: InferStudio downloads NOAA's published GFS forecast from the
same start time, for the same lead times and output variables, so it can be
compared with the AI models in every tab. GFS's archive starts in February 2021,
its forecasts go out to 384 h, and beyond 120 h its output is 3-hourly, so a
longer run needs an output frequency of 3 h or a multiple of it.

The download needs no GPU. With AI models selected it runs beside them in the
same job. On its own, the GPU settings are greyed out and it goes to a small
CPU-only job on Casper.

## Running several models at once

Several models in one job share the job's GPUs. With one GPU they run one after
another; with more, they are split between the GPUs (longest-running first) so
that no two models share a GPU at the same time, and run side by side. Each
extra GPU makes the job bigger, which can mean a longer wait in the queue.
