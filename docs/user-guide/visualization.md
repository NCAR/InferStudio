# Visualization

## Shared controls

Every plot in InferStudio is driven from one place. A single controls object owns
variable, level, lead time, colormap, and geographic extent; each plot panel
subscribes to it and re-renders when it changes.

The consequence worth internalizing: **you cannot put two panels into different
states.** Changing the variable changes it everywhere. This is the point — a
side-by-side comparison in which one panel is showing 500 hPa geopotential at
+48 h and the other is showing it at +54 h is worse than no comparison at all,
and a per-panel control scheme makes that mistake easy to commit and hard to
notice.

If you genuinely need divergent panels, take a screenshot of one state before
changing controls, or export the fields and plot them yourself.

## Variable and level

How these two controls behave depends on the convention of the active dataset:

:::{list-table}
:header-rows: 1
:widths: 30 35 35

* - Convention
  - Variable control
  - Level control
* - Earth2Studio (flattened)
  - Lists composite names: `u850`, `t500`, `z500`, `u100`
  - Hidden — the level is part of the variable name
* - ERA5-style (CREDIT / WXFormer)
  - Lists bare names: `U`, `T`, `Z`
  - Active, listing the dataset's vertical coordinate values
:::

This is why convention detection in the dataset browser matters: the control
layout is not cosmetic, it reflects the actual shape of the array being indexed.

## Stepping through lead time

The lead-time control moves all panels forward together. Two idioms:

- **Drag** to scrub, for finding a feature.
- **Step** with the arrow controls, for frame-by-frame comparison — this is
  usually how you spot the lead time at which two models diverge.

For Earth2Studio output, lead time is a real dimension and stepping is a cheap
array index. For ERA5-style output with separate time steps in separate files,
stepping may trigger a file read; expect a brief pause on the first pass through.

## Colormaps and scaling

The colormap applies globally. Two rules of thumb:

- Use a **diverging** map centred on zero for anomaly and difference fields, and
  a **sequential** map for absolute fields. A sequential map on a difference
  field will hide the sign.
- Keep the colour limits fixed when comparing models. Per-panel autoscaling makes
  a model with a large bias look identical to one without.

## Projections and regional extents

Setting an extent zooms every panel to the same window. Note the longitude
convention of the dataset before typing bounds — Earth2Studio output is commonly
0–360, so a North American window is roughly 230–300 rather than −130 to −60.

## Panel sizing

Plots are sized to fill their container. If a plot appears squashed to a few
pixels tall, this is a known layout interaction rather than a data problem:
`sizing_mode="stretch_both"` does not propagate reliably through Bokeh's tab
shadow DOM. The maintainers' fix is explicit pixel heights or wrapping the plot
function with an explicit `stretch_width` sizing mode; CSS rules in the app's
stylesheet override Python-side sizing where the two disagree. Reloading the page
after the tab is already rendered usually clears a transient case.

## Exporting

Bokeh's toolbar provides a PNG save on each panel. For publication figures, the
better path is to load the NetCDF yourself and plot with matplotlib or Cartopy —
InferStudio is a screening and comparison tool, not a figure-preparation tool.
