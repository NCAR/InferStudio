# Troubleshooting

Organized by what you observe, since the same underlying cause shows up
differently depending on where you are looking.

## The app will not load

**Blank page, or a spinner that never resolves.**
Through Open OnDemand, reload the page; if it persists, end the session and
launch a new one. When you serve the app yourself, it's almost always a WebSocket
problem: check the serving terminal for a rejected-origin message and set
`--allow-websocket-origin` to the host you reach the app through
(`localhost:5006` behind an SSH tunnel). A proxy that strips WebSocket upgrades
produces the same symptom with no server-side message.

**Connection refused (serving it yourself).**
The server is not running, the port does not match, or your tunnel points at the
wrong compute node. Confirm the node with `qstat -u $USER -n`.

**`ModuleNotFoundError` on startup.**
Check the capitalization of the module first — `earth2StudioPlot`, not
`earth2studioPlot`. If that is right, confirm you activated the Panel conda
environment rather than a per-model venv.

## The session ended

**The page says the connection was lost.**
Usually the session's wall time ran out; the header counts down to it and warns
15 and 5 minutes before. Inference jobs are separate PBS jobs and keep running.
Follow them from a Casper or Derecho login node:

```bash
qstat -u $USER @casper-pbs @desched1
```

Then start a new session and open the results with **Load Existing Suite**. For
a longer session, raise **Wall Time** on the launch form (up to 24 hours). See
{doc}`../getting-started/launching`.

**The session will not start; it sits queued.**
Sessions run on Casper's shared CPU nodes and normally start within minutes. If
one stays queued, Casper is busy; the session card shows its state, as does
`qstat -u $USER @casper-pbs`.

**Session ended early with no obvious error.**
Check the wall time first, then memory: a session has 32 GB, and PBS kills it if
it goes over. `qhist -j <jobid>` shows the accounting record; a *Used Mem* close
to 32 is worth reporting, along with what you were viewing.

## The inference job

**It sits queued.**
While the job waits, **Compute Settings** shows PBS's estimate of when it will
start. Derecho's *main* queue waits for a whole free node; its *develop* queue
shares nodes but allows only 6 hours. Another system or GPU type may start
sooner.

**A model's log ends with "The PBS job ended before this model finished".**
The job was stopped, most often by running past its wall time, which InferStudio
estimates from the models and forecast length. The message includes the end of
the job's own output. Report it, since it means the estimate was too short.

## A run fails

**Fails within seconds, data-source error or HTTP 404.**
The initialization time has no published analysis. Move back one or two 6-hourly
cycles. Remember the picker snaps to 00/06/12/18 UTC, so the time in the log may
not be the time you typed.

**Fails on import of a compiled extension (`undefined symbol`).**
An environment problem, not a usage problem. The extension was built against a
different PyTorch than the one installed in that venv. See
{doc}`../models/environments`.

**`no kernel image is available for execution on the device`.**
The extension was compiled without the compute capability of the GPU the job
landed on. Rebuild for every GPU InferStudio offers:
`TORCH_CUDA_ARCH_LIST="7.0;8.0;8.9;9.0"` (V100, A100, L40, H100).

**`CUDA driver version is insufficient`.**
PyTorch in that venv is built against a CUDA newer than the node's driver
supports. Something upgraded torch — usually Earth2Studio's resolver. Re-pin.

**Out of GPU memory.**
Each model runs on a GPU of its own, so asking for more GPUs won't help. The log
names a larger GPU type to choose under **Compute Settings**; AIFS, for one,
needs an 80 GB card.

**Runs but is inexplicably slow (SFNO or FourCastNet 3).**
`makani` may have silently fallen back to a non-CUDA path because a git-installed
`torch-harmonics` reports version `0.0.0` and fails its extension-availability
check. See the FCN3 notes in {doc}`../models/environments`.

## Output problems

**The run says complete but the dataset browser shows nothing.**
Confirm the output directory in the log's completion message, and confirm the
browser is pointed at the same place. Scratch purges can also remove older output
between sessions.

**The CF-compliant copy is missing.**
The log says `CF conversion failed, raw output is still available`; the model's
own NetCDF file is complete and can be loaded as usual.

**Wrong plot controls — no level selector when you expect one.**
The dataset's convention was detected as Earth2Studio because it has a `lead_time`
dimension. See {doc}`output-formats`.

**Variable list is empty.**
The file opened but nothing matched the expected structure. Inspect it directly:

```bash
ncdump -h output.nc | head -50
```

## UI problems

**The interface froze.**
This is the signature of a threading deadlock: a background thread mutating a
Bokeh or Panel widget directly instead of scheduling the update through the
document's next-tick callback. If you hit it in current code, it is a bug worth
reporting. Any inference job keeps running regardless; reload the page, or
start a new session and open the results with **Load Existing Suite** once the
job is done.

**A plot is a few pixels tall.**
`sizing_mode="stretch_both"` does not propagate through Bokeh's tab shadow DOM.
Reloading the page after the tab has rendered usually clears it. The structural fix
is explicit pixel heights or a `stretch_width` wrapper.

**A slider will not render (Bokeh error E-1021).**
A zero-width range — an integer slider whose start equals its end. Occurs when a
dataset has a single lead time or a single level.

**Duplicate notification toasts.**
A known artifact of notebook-embedded Panel firing notifications twice; deduplication
is in place, but if you see it, note the action that triggered it in your report.

## Getting help

Before opening an issue, collect:

1. The failing model's log: its log tab, or
   `<output directory>/<simulation name>/<model>/<model>_run.log`
2. The PBS job ID, shown under **Launcher**, and the job's own output,
   `pbs.out` in its folder under `~/.inferstudio/jobs/`
3. The model, initialization time, and lead time
4. The system and GPU type the job ran on
5. Whether the same run succeeds with that model selected alone

Use **Questions & Feedback** in the app's header, or open an issue at
<https://github.com/NCAR/InferStudio/issues>.
