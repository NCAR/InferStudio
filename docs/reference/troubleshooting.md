# Troubleshooting

Organized by what you observe, since the same underlying cause shows up
differently depending on where you are looking.

## The app will not load

**Blank page, or a spinner that never resolves.**
Almost always a WebSocket problem. Check the serving terminal for a rejected-origin
message and set `--allow-websocket-origin` to the host you are actually reaching
the app through (the Hub hostname behind JupyterHub, `localhost:5006` behind an SSH
tunnel). A proxy that strips WebSocket upgrades produces the same symptom with no
server-side message.

**Connection refused.**
The server is not running, the port does not match, or your tunnel points at the
wrong compute node. Confirm the node with `qstat -u $USER -n`.

**`ModuleNotFoundError` on startup.**
Check the capitalization of the module first — `earth2StudioPlot`, not
`earth2studioPlot`. If that is right, confirm you activated the Panel conda
environment rather than a per-model venv.

## The job disappeared

**The whole session vanished after an SSH drop.**
An interactive `qsub -I` session is tied to your connection. Use `tmux` on a
specific login node before allocating, or a holder job — see
{doc}`../getting-started/launching`.

**The job will not start; it sits queued.**
The `vis` queue is capped at three concurrent jobs across the system and is
frequently full. Check with `qstat -q vis`. Use `nvgpu` instead, which usually has
headroom.

**Job terminated with no obvious error.**
Check walltime first — an exceeded walltime kills the job with little ceremony. Then
memory. `qhist -j <jobid>` shows the accounting record.

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
The extension was compiled without the L40's compute capability. Rebuild with
`TORCH_CUDA_ARCH_LIST="8.9"`.

**`CUDA driver version is insufficient`.**
PyTorch in that venv is built against a CUDA newer than Casper's 12.7 driver
ceiling. Something upgraded torch — usually Earth2Studio's resolver. Re-pin.

**Out of memory, but only when running several models.**
Concurrent models share one GPU by default. Either request more GPUs, set
`CUDA_VISIBLE_DEVICES` per model, or run sequentially.

**Runs but is inexplicably slow (SFNO or FourCastNet 3).**
`makani` may have silently fallen back to a non-CUDA path because a git-installed
`torch-harmonics` reports version `0.0.0` and fails its extension-availability
check. See the FCN3 notes in {doc}`../models/environments`.

## Output problems

**The run says complete but the dataset browser shows nothing.**
Confirm the output directory in the log's completion message, and confirm the
browser is pointed at the same place. Scratch purges can also remove older output
between sessions.

**Wrong plot controls — no level selector when you expect one.**
The dataset's convention was detected as Earth2Studio because it has a `lead_time`
dimension. See {doc}`output-formats`.

**Variable list is empty.**
The file opened but nothing matched the expected structure. Inspect it directly:

```bash
ncdump -h output.nc | head -50
```

## UI problems

**The interface froze but the inference is still running.**
This is the signature of a threading deadlock: a background thread mutating a
Bokeh or Panel widget directly instead of scheduling the update through the
document's next-tick callback. If you hit it in current code, it is a bug worth
reporting with `/tmp/debug.log` attached — capture the log before starting another
run, since it is truncated at each run start.

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

1. The full log tab contents for the failing model
2. `/tmp/debug.log`, captured before any subsequent run
3. The model, initialization time, and lead time
4. `qstat -f $PBS_JOBID` output if the failure is job-related
5. Whether the same run succeeds with that model selected alone

Issues go to <https://github.com/NCAR/InferStudio/issues>.
