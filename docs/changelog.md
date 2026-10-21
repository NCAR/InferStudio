# Changelog

Notable changes to InferStudio. Newest first.

:::{note}
This file is a starting point assembled from the recent development history. Fill
in real dates and version numbers, and prune anything that is not user-visible —
a changelog that lists internal refactors alongside features stops being useful
for deciding whether to upgrade.
:::

## Unreleased

### Added

- DLESyM v0 (ISCCP-ERA5) on the Climate tab, beside CAMulator. The start date
  is now a year and month that every selected model can start from; Z500,
  T850, TCWV and SST can be saved; and a variable a model lacks is skipped
  with a warning.

- A spinner next to a tab's title while it's computing: Visualization while a
  dataset loads, a difference computes or a video exports; Statistics (and
  its Basic or credit-verify sub-tab) while stats compute; Inference (and its
  Weather or Climate sub-tab) while a run is going.

- Physical models for comparison: GFS can be selected on the Weather tab. Its
  published forecast for the same start time is downloaded into the suite,
  with no GPU, and shows up in plots, differences and statistics like the AI
  models.

- Inference runs in its own PBS batch job instead of on the session's node. The
  Inference tab's new Compute Settings section picks Derecho (the default) or Casper, the GPU
  type (Casper's A100 80GB by default) and the number of GPUs; the wall time is
  estimated, and the job ends when the last model finishes. The Launcher shows the time spent in the queue separately from
  the elapsed run time.
- The Statistics tab has a credit-verify sub-tab beside the existing Basic one:
  WeatherBench2-style latitude-weighted RMSE and bias against GFS analysis,
  globally and for the tropics and extratropics, using credit-verify's scoring
  code.
- The Visualization tab's time slider tooltip shows the selected step's valid
  time (e.g. 2026-10-19 12:00 UTC) instead of its index.
- Forecast statistics panel: RMSE, CRPS, and spread/error ratio computed against a
  GFS reference.
- Earth2Studio output plotting via `earth2StudioPlot.py`, handling the flattened
  level-variable naming convention.
- Per-model log tabs with live status indicators (`◷` / `⟳` / `✓` / `✗`), elapsed
  time labels, and completion path messages.
- Automatic simulation-name generation from model and initialization time.
- NSF NCAR branding.
- Lead-time slider in the time picker, replacing the earlier absolute-time entry.

### Changed

- Plot state consolidated into a single shared controls object driven from the
  sidebar; all plots now update together.
- Dataset scanner detects the `lead_time` dimension and branches between
  Earth2Studio and ERA5-style handling instead of assuming one convention.
- Dataset browser enforces single selection, with checkbox state and selection kept
  in two-way sync.
- Adding datasets appends to the existing list rather than rebuilding it.
- Default output path is `/glade/scratch/$USER`.

### Fixed

- A model run that died part way no longer breaks the plots. Its unwritten
  steps hold netCDF's default fill value (9.97e+36), which was read as data and
  stretched the shared colorbar so every model looked flat. They're now treated
  as missing in plots, differences and statistics.

- The Models card on the Inference tab is indented to line up with the other
  cards.

- Model output is CF-converted again. The job scripts imported `cf_convert`
  from `/glade/work/pearse/InferStudio`, which no longer exists, so recent runs
  only had their raw output.

- Threading deadlock caused by background threads mutating Bokeh and Panel widgets
  directly; updates are now scheduled through the document's next-tick callback.
- Bokeh E-1021 zero-width slider error for datasets with a single lead time or
  level.
- Duplicate notification toasts when the app is embedded in a notebook context.
- Plot panels collapsing to near-zero height inside Bokeh `Tabs`.

## Earlier

Foundational work: Panel application architecture, the abstract `ModelRunner` base
with Earth2Studio and MILES-CREDIT implementations, GDAS cycle snapping,
subprocess-based inference with streaming output and process-group cancellation,
and the per-model uv environment scheme for Pangu, AIFS, Aurora, SFNO, and
FourCastNet 3.
