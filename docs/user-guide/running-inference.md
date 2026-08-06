# Running inference

## What happens when you press Run

InferStudio does not import the models into its own process. Doing so would be
impossible — the dependency stacks conflict — so instead each run is a
subprocess launched with a *different Python interpreter*.

```text
Panel app  (conda: creditJun3)
    │
    ├─ subprocess.Popen(  /glade/work/$USER/E2S/envs/pangu/bin/python  -c "<script>" )
    ├─ subprocess.Popen(  /glade/work/$USER/E2S/envs/sfno/bin/python   -c "<script>" )
    └─ subprocess.Popen(  <credit env>/bin/python  ... )               # WXFormer
             │
             └─ stdout/stderr ──▶ streamed into the model's log tab
```

The mapping from model to interpreter lives in a `MODEL_ENV_MAP` dictionary in
the Earth2Studio runner module. The generated script is what actually calls
`earth2studio.run.deterministic()` (or the ensemble equivalent) with the model
package, the data source, and the IO backend.

Each process is started in a new session so that **Cancel** can signal the whole
process group. Killing only the parent would orphan the CUDA-holding child and
leave the GPU occupied.

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

**Cancel** sends a termination signal to the process group. Partial output may
remain in the output directory; delete it before re-running with the same
simulation name, or pick a new name.

Re-running with an identical name overwrites. The auto-generated names include
the model and initialization time, which makes accidental collisions unlikely but
not impossible when you iterate on the same case.

## Running several models at once

Selecting multiple models fans out into concurrent subprocesses on the same node.
This is the intended comparison workflow, but be aware of the resource
arithmetic: several models on one GPU will contend for memory. If you see
out-of-memory errors that do not occur when models run alone, either request more
GPUs in your allocation or run the models sequentially.
