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
