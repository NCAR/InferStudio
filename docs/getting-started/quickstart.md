# Quickstart

This walks through one deterministic forecast end to end: pick a model, run it,
look at the output, and score it. It assumes the app is already open (see
{doc}`launching`).

## 1. Name the run

The **Simulation name** field at the top of the sidebar is pre-filled with an
auto-generated name derived from the model and initialization time. Accept it or
type your own — it becomes the output subdirectory name, so keep it filesystem
safe.

## 2. Choose one or more models

Select from the model list. Each selected model gets:

- its own inference process, dispatched into its own environment
- its own log tab in the output pane
- its own status indicator (`◷` queued, `⟳` running, `✓` complete, `✗` failed)

Selecting more than one model is the normal comparison workflow — they share the
initialization time and lead time, which is what makes the later scoring
meaningful.

## 3. Set the initialization time

The time picker takes a date and hour. Earth2Studio pulls initial conditions
from GFS/GDAS analyses, which exist only at the synoptic cycles, so **the picker
snaps your selection to the nearest 00, 06, 12, or 18 UTC cycle**. If you enter
`09:40`, it resolves to `06:00`.

:::{warning}
Very recent dates may not have an analysis published yet, and very old dates may
have aged out of the fast archive. If a run fails immediately with a data-source
error, move the initialization back by a cycle or two.
:::

## 4. Set the lead time

The lead-time slider is expressed in forecast hours and is stepped at the
model's native output interval (6 h for most Earth2Studio models). Longer leads
cost proportionally more GPU time and disk.

## 5. Set the output directory

Defaults to `/glade/scratch/$USER`. Output lands in
`<output-directory>/<simulation-name>/`.

:::{important}
GLADE scratch is purged on a rolling schedule. Move anything you intend to keep
into your work or campaign space.
:::

## 6. Run

Press **Run**. Watch the log tab for the selected model — it streams the child
process's stdout and stderr live, including model download progress on a first
run. An elapsed-time label ticks alongside the status glyph, and the full output
path is printed on completion.

**Cancel** terminates the process group, not just the parent shell, so a
half-finished inference does not keep holding the GPU.

## 7. Load the output

Open the dataset browser and select the run you just produced. The browser
inspects the file and detects which convention it follows — the presence of a
`lead_time` dimension marks it as Earth2Studio output rather than an ERA5-style
CREDIT file — and configures the plotting controls accordingly.

Dataset selection is **single-select** by design: the plot controls are driven
from one active dataset at a time, with comparisons handled by the plot panel
rather than by multi-select.

## 8. Plot

The sidebar plot controls (variable, level, lead time, colormap, projection)
drive every open plot simultaneously. Step the lead-time control to animate
through the forecast; all panels stay in sync.

## 9. Score it

Open the forecast statistics panel. It computes, against a GFS reference for the
same valid times:

- **RMSE** per lead time
- **CRPS**, for ensemble output
- **Spread/error ratio**, as a calibration check

## Where to go next

- {doc}`../user-guide/interface` — every control, described
- {doc}`../models/index` — what each model expects and produces
- {doc}`../reference/output-formats` — reading the NetCDF yourself
