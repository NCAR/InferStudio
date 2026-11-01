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

- Each converted model file (`*_cf.nc`) records, in its global attributes,
  every change the conversion made to the data (`conversion_steps`) and the
  run that produced it: the model and its version, the framework and its
  version, initial conditions, times, GPU and PBS job (`inference_*`), plus
  CF's `title`, `source` and `history`.

- The examples shown at startup are now WeatherExample (every weather model,
  from the previous day's analysis) and ClimateExample (every climate model),
  rerun every night with their model differences and statistics already
  computed.

- CESM2-LE as a physical model on the Climate tab: one historical member's
  6-hourly CAM6 output (1850-2014) read from GDEX in CAMulator's layout, to
  compare with CAMulator and DLESyM. A member picker appears under Time
  Settings when it's selected.

- Casper's V100 (32 GB) GPUs under Compute Settings, with a warning that AIFS,
  Aurora and FourCastNet3 are likely to run out of memory on them.

- Video export layout: by default simulations are now arranged in whichever
  grid fills the chosen resolution best (e.g. 2×2 rather than a tall single
  column), instead of mirroring the screen's one per row; a Layout menu sets
  1–4 per row by hand. The ▲/▼ arrows beside each simulation set its order in
  the video, and a preview shows the frame — with any empty space in grey —
  before exporting.

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

- WeatherExample is now a 10-day forecast (41 steps of 6 hours), not 18
  hours. Its statistics cover the steps whose valid time has passed; the
  Statistics tab likewise skips steps still in the future, which have no
  analysis to verify against.

- The Metadata section's "Forecasts" row is now "NumTS", under Ts End, as
  the Time section labels it.

- The colormaps are VAPOR's 34 built-in ones, computed as VAPOR computes them
  (Moreland's interpolation for the diverging ones), in one list of swatches
  with the sequential ones first, and with Reverse. Kindlmann,
  KindlmannExtended and BlackBody use Kenneth Moreland's published tables
  instead of VAPOR's control points, which showed bands. Before,
  only six matplotlib maps were offered, since cmocean wasn't installed. The
  selected map's name is shown above its swatch, where it no longer runs out
  of the sidebar. Differences use VAPOR's CoolWarm.

- WXFormer is always the top row of the Visualization tab.

- Usage logs moved to `/glade/campaign/cisl/vast/pearse/inferStudio/logs`,
  and the nightly example jobs' logs to its `nightly/` subdirectory.

- WXFormer (gen 1), MILES-CREDIT's original pipeline, is disabled on the
  Weather tab: its output can't be shown. WXFormer (gen 2) runs the same
  weights.
- Plot state consolidated into a single shared controls object driven from the
  sidebar; all plots now update together.
- Dataset scanner detects the `lead_time` dimension and branches between
  Earth2Studio and ERA5-style handling instead of assuming one convention.
- Dataset browser enforces single selection, with checkbox state and selection kept
  in two-way sync.
- Adding datasets appends to the existing list rather than rebuilding it.
- Default output path is `/glade/scratch/$USER`.

### Fixed

- A time step a model didn't write (WXFormer's first, its initial state)
  showed as a collapsed, empty sliver; it's now a full-size blank panel with
  "No Data" across it. A suite with WXFormer opens on its first step with
  data.

- WXFormer's panel drew as a narrow sliver with no title or coastlines even
  where it had data: its uneven (Gaussian) latitudes made it a QuadMesh,
  which lost its plot options when rasterized. It's now interpolated onto an
  even grid of as many latitudes first.

- Global maps showed the world three times side by side; they now open on one
  globe, from -180 to 180 whatever longitudes the model wrote, and still pan
  across the dateline.

- Choosing a dataset shows the loading spinner straight away. Before, the
  outgoing dataset reloaded all its plots for the new selection's variable
  first, which could take several seconds with no sign anything was happening.

- Differences between two models on the same grid written with slightly
  different coordinates (CAMulator and CESM2-LE's latitudes differ in the
  last digit) covered only about half the grid; the rest was silently left
  out.

- FourCastNet3 output now shows in the Visualization tab, with differences
  against the other models; it said plotting "isn't wired up yet".

- WXFormer output now shows in the Visualization tab and Statistics: when a run
  ends, its per-lead-time files are gathered into one file in the other weather
  models' layout. It has 500, 700 and 850 hPa only, and its first (initial)
  time step is empty.

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
