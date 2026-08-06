# Interface tour

InferStudio uses a fixed two-region layout: a persistent **sidebar** that holds
every control, and a **main area** of tabs that holds output.

```text
┌──────────────────────┬──────────────────────────────────────────────┐
│  SIDEBAR             │  MAIN AREA                                   │
│                      │  ┌────────┬──────────┬─────────┬──────────┐  │
│  Simulation name     │  │Inference│ Datasets │  Plots  │  Stats   │  │
│  Model selection     │  └────────┴──────────┴─────────┴──────────┘  │
│  Initialization time │                                              │
│  Lead time           │   Per-model log tabs, dataset browser,       │
│  Output directory    │   synchronized plots, score tables           │
│  [ Run ] [ Cancel ]  │                                              │
│  ──────────────      │                                              │
│  Plot controls       │                                              │
│   variable / level   │                                              │
│   lead time          │                                              │
│   colormap           │                                              │
└──────────────────────┴──────────────────────────────────────────────┘
```

The sidebar is deliberately the single source of truth. Plot state is not stored
per-tab; a `SharedPlotControls` object owns it and every plot subscribes, so
changing the variable updates all open panels at once. This is what makes
side-by-side model comparison honest — you cannot accidentally leave one panel
on a different level or lead time.

## Sidebar

### Run configuration

Simulation name
: Free text, pre-filled with an auto-generated name from the model and
  initialization time. Becomes the output subdirectory. Updates are committed on
  <kbd>Enter</kbd> or focus loss rather than on every keystroke, so typing does
  not trigger a cascade of re-renders.

Model selection
: One or more models. See {doc}`../models/index` for the roster.

Initialization time
: Date plus hour, snapped to the nearest 00/06/12/18 UTC analysis cycle.

Lead time
: A slider in forecast hours, stepped at the model's output interval.

Output directory
: Defaults to `/glade/scratch/$USER`.

Run / Cancel
: **Run** launches one child process per selected model. **Cancel** terminates
  the whole process group.

### Plot controls

These appear once a dataset is loaded, and are populated from that dataset's
actual contents rather than a hardcoded list.

| Control | Behaviour |
| --- | --- |
| Variable | Populated from the dataset. For Earth2Studio output the level is folded into the name (`u850`); for ERA5-style output variable and level are separate controls. |
| Level | Shown only when the dataset has a real vertical coordinate. |
| Lead time | Steps through forecast hours. Drives all open plots. |
| Colormap | Applies to every panel, so the comparison is visually fair. |
| Projection / extent | Regional zoom, applied consistently across panels. |

## Main area tabs

### Inference

Holds one log tab per selected model, plus a status row. Status is rendered as a
small set of glyphs:

| Glyph | Meaning |
| --- | --- |
| `◷` | Queued — process created, not yet producing output |
| `⟳` | Running — accompanied by a live elapsed-time label and a spinner |
| `✓` | Complete — the output path is printed in the log |
| `✗` | Failed — the traceback is in the log tab |

The view switches automatically to whichever tab most recently became active, so
you see the model that is actually doing something without hunting for it.

### Datasets

The dataset browser. Walks the output directory, groups by run, and shows the
metadata it can read from each file: variables, dimensions, coordinate ranges,
and which output convention was detected. Selection is single-select, with the
checkbox state and the underlying selection kept in two-way sync.

### Plots

Map panels for the active dataset, driven entirely from the sidebar. When
several models are loaded, panels are arranged for direct comparison at matched
variable, level, and lead time.

### Statistics

RMSE, CRPS, and spread/error ratio against a GFS reference. See
{doc}`statistics`.

## Notifications

Transient messages (run started, run finished, file not found) appear as toasts
in the corner. They are deduplicated, so a single event does not produce two
identical toasts when the app is embedded in a notebook context.
