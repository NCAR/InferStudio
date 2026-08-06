# Browsing and loading datasets

The **Datasets** tab is how output gets from disk into the plotting and scoring
panels. It is not limited to output produced in this session — you can point it
at any run in a readable directory, including runs produced outside InferStudio.

## Scanning

Give the browser a directory and it walks it, groups files by run, and inspects
each one. For every dataset it reports:

- the variables present
- the dimensions and their sizes
- coordinate ranges (time, `lead_time`, latitude, longitude, level)
- the **detected convention** — Earth2Studio flattened output or ERA5-style
  CREDIT output

Convention detection is the important step. It is driven by structure rather than
filename: the presence of a `lead_time` dimension identifies Earth2Studio output,
and the plotting controls branch on the result. Getting this wrong would mean
offering a level selector for a dataset whose levels are baked into the variable
names, or vice versa.

See {doc}`../reference/output-formats` for what each convention looks like.

## Selection

Selection is **single-select**. Only one dataset is active at a time, and the
sidebar plot controls are populated from it.

This is a deliberate design choice rather than a limitation. Comparison happens
at the *plot* level — several panels sharing one set of controls — not by loading
several datasets into one ambiguous control set. If two datasets disagreed about
which variables exist, a multi-select control panel would have to either take the
intersection (hiding data) or the union (offering controls that fail for one
dataset).

The checkbox UI and the underlying selection are kept in two-way sync, so
clicking a row and clicking its checkbox are equivalent, and programmatic
selection changes are reflected in the checkboxes.

## Adding datasets incrementally

Adding a directory **appends** to the browser rather than replacing it. Previously
scanned datasets stay listed. This lets you build up a working set — a few of
your own runs plus a reference archive — without re-scanning everything each
time.

## Metadata inspection

Selecting a dataset exposes its metadata without plotting anything. This is often
all you need: to confirm a run actually reached the lead time you asked for, to
check that a variable you expect is present, or to see whether longitude runs
0–360 or −180–180 before you write a regional extent.

## Loading external data

Two categories of file are commonly loaded alongside forecast output:

Reference analyses
: GFS or ERA5 fields for the same valid times, used as truth in the statistics
  panel.

Other groups' output
: Any CF-ish NetCDF that follows one of the two supported conventions will load.
  Files that follow neither may still open in the metadata view but will not
  populate the plot controls correctly.

:::{note}
Earth2Studio's built-in IO backends (netcdf4, xarray, zarr) do not emit fully
CF-compliant metadata as of v0.16.0. InferStudio reads its own output fine, but
if you intend to hand a file to a CF-strict tool, run it through a
post-processing step first. See {doc}`../reference/output-formats`.
:::
