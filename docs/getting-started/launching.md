# Launching InferStudio

InferStudio runs as an interactive app on
[NCAR's Open OnDemand portal](https://ondemand.hpc.ucar.edu/). The session that
serves the app runs on Casper CPUs only. When you press **Run Inference**,
InferStudio submits the models as a separate PBS job, with the GPUs you choose
on the Inference tab.

## Starting a session

1. Sign in to <https://ondemand.hpc.ucar.edu/> and open **InferStudio** under
   **Interactive Apps**.
2. Fill in the form:

   **Resource Selection**
   : *Casper PBS batch*, the only option.

   **Project Account**
   : The project charged for the session, and for every inference job it
     submits, on Casper or Derecho.

   **Wall Time**
   : How long the session lasts: 2 hours by default, up to 24.

   **Conda Environment**
   : Leave it on *InferStudio* unless you've been told otherwise.

3. Press **Launch**. When the session card shows *Running*, press
   **Connect to InferStudio**.

Every session gets 2 CPUs and 32 GB of memory, and no GPU. That's enough for
browsing and plotting results; the models themselves never run in the session.

## GPUs for inference

The GPUs are chosen per run, under **Compute Settings** on the Inference tab:
Casper or Derecho, the GPU type, and how many. InferStudio estimates the job's
wall time from the models and the forecast length, and while the job is queued
it shows PBS's estimate of when it will start.

## When the session ends

The page header counts down to the end of the session's wall time and warns 15
and 5 minutes before it.

Inference jobs are separate from the session, so they keep running after it
ends. To follow them, run this on a Casper or Derecho login node; their names
start with `InferStudio`:

```bash
qstat -u $USER @casper-pbs @desched1
```

When a job finishes, start a new session and open its results with
**Load Existing Suite**.

## Running it outside Open OnDemand

For development, you can serve the app yourself from a Casper compute node.

**Step 1: get a node.** From a Casper login node:

```bash
qsub -I -q casper -l select=1:ncpus=2:mem=32GB \
     -l walltime=02:00:00 -A <PROJECT_CODE>
```

Note the node you land on (`crhtc52`, for example).

**Step 2: start the server.**

```bash
module load ncarenv/25.10 cuda/12.9.0 conda
conda activate /glade/work/pearse/conda-envs/InferStudio
cd <your InferStudio checkout>
panel serve panel_app.py --port 5006 --allow-websocket-origin=localhost:5006
```

Inference jobs are charged to the project of the PBS job the server runs in.
Outside a PBS job, set `INFERSTUDIO_ACCOUNT=<PROJECT_CODE>` before starting
the server, or **Run Inference** will refuse to submit. There's no countdown
in the header, since the app isn't told when the session ends.

**Step 3: forward the port from your laptop**, through the login node:

```bash
ssh -N -L 5006:crhtc52:5006 <username>@casper.hpc.ucar.edu
```

**Step 4:** open <http://localhost:5006/panel_app> in your browser.

:::{note}
`--allow-websocket-origin` is required. Panel rejects WebSocket connections
whose `Origin` header doesn't match the serving host, and through a tunnel or
proxy the origin isn't the compute node.
:::

## Verifying the launch

A healthy start prints a Bokeh server line and then goes quiet:

```text
Launching server at http://localhost:5006
```

If the browser shows a blank page or a spinner that never resolves, check the
terminal for a `WebSocket connection ... rejected` message. That is almost
always an `--allow-websocket-origin` mismatch rather than a problem with the
application.

See {doc}`../reference/troubleshooting` when a job disappears or the browser
tab goes stale.
