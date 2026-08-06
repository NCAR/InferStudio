# Prerequisites

InferStudio is not a standalone desktop application. It runs *on* an NSF NCAR
compute node and is viewed in your local browser, so you need HPC access before
anything else.

## Accounts and allocations

| Requirement | Notes |
| --- | --- |
| NSF NCAR HPC account | Active login for the Casper cluster, with multifactor authentication set up. |
| Project (account) code | Charged for the GPU node hours. Runs are submitted against a code such as `NVST0001` or `UCIS0005`. |
| GLADE file space | Write access to a scratch or work directory for forecast output. |

:::{tip}
Your available project codes are listed in the NSF NCAR Systems Accounting
Manager portal. To see what you have actually been charging against recently:

```bash
qhist -u $USER -f account,queue,reqmem,elapsed
```
:::

## Compute resources

Inference is GPU work. On Casper the relevant queues are:

:::{list-table}
:header-rows: 1
:widths: 18 22 60

* - Queue
  - Hardware
  - Notes
* - `vis`
  - L40 nodes (`casper47`–`casper50`)
  - 48 CPUs, 7 GPUs, ~750 GB RAM per node. **Capped at 3 concurrent jobs**, so
    this queue is frequently saturated.
* - `nvgpu`
  - Mixed NVIDIA GPU nodes
  - Usually has more headroom than `vis`; prefer it for long or batch runs.
* - `casper`
  - Routing queue
  - Submissions are routed to an execution queue; useful when you do not care
    which one you land in.
:::

A single deterministic global forecast at 0.25° needs one GPU and fits
comfortably in a `vis` or `nvgpu` allocation. Ensemble runs and CRPS scoring
scale with member count, so request walltime accordingly.

## Software already provided

You do **not** need to install Earth2Studio, PyTorch, `makani`, or
`torch-harmonics` yourself. InferStudio ships with a set of pre-built,
per-model environments and dispatches each run into the correct one. See
{doc}`../models/environments` for the layout, and for what to do when an
environment is missing or broken.

The Panel application itself runs from a conda environment maintained alongside
the code:

```text
Application code:   /glade/work/<user>/InferStudio/
Panel environment:  /glade/work/<user>/conda-envs/creditJun3/
Per-model venvs:    /glade/work/<user>/E2S/envs/<model>/
```

:::{admonition} Site-specific paths
:class: important

The paths above reflect a single developer installation. If your group runs a
shared deployment, substitute the shared prefix. Ask the maintainers for the
canonical location rather than cloning your own copy — the per-model
environments are expensive to rebuild.
:::

## Local requirements

- A modern browser (Chrome, Firefox, or Safari). Panel relies on WebSockets;
  proxies that strip WebSocket upgrade requests leave the app visibly loaded but
  permanently unresponsive.
- An SSH client, if you plan to launch outside of JupyterHub.
