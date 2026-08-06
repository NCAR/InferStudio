# Launching InferStudio

There are two supported ways to get the app in front of you. The JupyterHub
route is the one to use unless you have a reason not to.

## Option 1 — JupyterHub on Casper (recommended)

JupyterHub already handles the authentication, the compute-node allocation, and
the proxying, so no manual tunnel is needed.

1. Sign in to the NSF NCAR JupyterHub and start a **Casper** server. Request a
   GPU resource — the app and the inference share the node.
2. Open a terminal in JupyterLab.
3. Activate the Panel environment and start the server:

   ```bash
   module load conda
   conda activate /glade/work/$USER/conda-envs/creditJun3
   cd /glade/work/$USER/InferStudio
   panel serve app_layout.py --port 5006 --allow-websocket-origin='*'
   ```

4. Open the app through the JupyterHub proxy path:

   ```text
   https://<jupyterhub-host>/user/<your-username>/proxy/5006/app_layout
   ```

:::{note}
`--allow-websocket-origin` is required. Panel rejects WebSocket connections whose
`Origin` header does not match the serving host, and behind the Hub proxy the
origin is the Hub, not the compute node. Pinning it to the Hub hostname instead
of `'*'` is tighter, and worth doing for a shared deployment.
:::

## Option 2 — Manual SSH port forward

Use this when you want the app on a node you allocated yourself, or when the Hub
is unavailable.

**Step 1 — get an interactive GPU allocation.** From a Casper login node:

```bash
qsub -I -q vis -l select=1:ncpus=8:ngpus=1:mem=64GB \
     -l walltime=04:00:00 -A <PROJECT_CODE>
```

Note the compute node you land on (`casper49`, for example) — you need it in
step 3.

**Step 2 — start the server on the compute node:**

```bash
module load conda
conda activate /glade/work/$USER/conda-envs/creditJun3
cd /glade/work/$USER/InferStudio
panel serve app_layout.py --port 5006 --allow-websocket-origin=localhost:5006
```

**Step 3 — forward the port from your laptop.** Jump through the login node to
the compute node in one hop:

```bash
ssh -N -L 5006:casper49:5006 <username>@casper.hpc.ucar.edu
```

**Step 4** — open <http://localhost:5006/app_layout> in your browser.

## Keeping the session alive

An interactive `qsub -I` session dies with your SSH connection, which takes the
Panel server and any running inference with it. Two habits prevent losing a
long run:

`tmux` on the login node
: Start `tmux` on a login node *before* you `qsub -I`. When the connection
  drops, reconnect and `tmux attach`. Pick a specific login node (for example
  `crlogin2`) and always come back to that one — `tmux` sessions are per-host,
  and the round-robin alias will otherwise land you somewhere else.

A "holder" job
: For work spanning many hours, submit a placeholder batch job that reserves
  the node, then attach your interactive shell to it. A minimal script:

  ```bash
  #!/bin/bash
  #PBS -N holder
  #PBS -q vis
  #PBS -A <PROJECT_CODE>
  #PBS -l select=1:ncpus=48:ngpus=1:mem=700GB:place=excl
  #PBS -l walltime=12:00:00
  sleep 12h
  ```

  `place=excl` keeps other jobs off the node so your GPU memory is not
  contended. Remember that the `vis` queue's three-job cap counts holder jobs.

See {doc}`../reference/troubleshooting` when a job disappears or the browser tab
goes stale.

## Verifying the launch

A healthy start prints a Bokeh server line and then goes quiet:

```text
Launching server at http://localhost:5006
```

If the browser shows a blank page or a spinner that never resolves, check the
terminal for a `WebSocket connection ... rejected` message — that is almost
always an `--allow-websocket-origin` mismatch rather than a problem with the
application.
