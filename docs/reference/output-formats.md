# Output formats

InferStudio reads two NetCDF conventions. You need to know which one you have
before writing any analysis code against a file, because the same physical field
is indexed differently in each.

## Earth2Studio (flattened)

Produced by all Earth2Studio models.

**Distinguishing feature:** a `lead_time` dimension, and variable names with the
pressure level folded in.

```text
dimensions:
    time      = 1        # initialization time
    lead_time = 41       # forecast steps
    lat       = 721
    lon       = 1440

variables:
    float32 u850(time, lead_time, lat, lon)     # u wind at 850 hPa
    float32 t500(time, lead_time, lat, lon)     # temperature at 500 hPa
    float32 z500(time, lead_time, lat, lon)
    float32 u100(time, lead_time, lat, lon)     # u wind at 100 hPa
    float32 t2m(time, lead_time, lat, lon)      # 2 m temperature (surface)
```

Reading it:

```python
import xarray as xr

ds = xr.open_dataset("output.nc")
field = ds["z500"].isel(time=0, lead_time=8)   # +48 h if 6-hourly output
```

Notes:

- `lead_time` is typically a `timedelta64`. Convert with
  `ds.lead_time / np.timedelta64(1, "h")` for plotting against forecast hours.
- Level is **not** a coordinate. There is no way to select "all levels of u" with
  a single index; you select variables by name.
- `u100` is 100 **hPa**, not 100 **metres**. This is a real ambiguity in the
  flattened naming scheme and a genuine source of error — confirm against the
  model's variable list rather than assuming.
- Longitude commonly runs 0–360.

## ERA5-style (ERA5, raw WXFormer)

The shape of ERA5 itself, and of what MILES-CREDIT's WXFormer saves: one file
per lead time, `<init>Z/pred_<init>Z_<FHR>.nc` in its model directory. The
Visualization tab doesn't read those; when the run ends, InferStudio gathers
WXFormer's pressure-level and surface fields from them into one
`<name>_WXFormerGen2_cf.nc`, in the same CF layout and variable names (`u`, `t`,
`sp`, `msl`, ...) as the other weather models. It saves 500, 700 and 850 hPa
only, and no initial state: its first time step is NaN.

**Distinguishing feature:** a real vertical coordinate, bare variable names, no
`lead_time` dimension.

```text
dimensions:
    time  = 41          # valid times
    level = 13
    lat   = 721
    lon   = 1440

variables:
    float32 U(time, level, lat, lon)
    float32 T(time, level, lat, lon)
    float32 Z(time, level, lat, lon)
    float64 level(level)                # pressure, hPa
```

Reading it:

```python
ds = xr.open_dataset("output.nc")
field = ds["Z"].sel(level=500).isel(time=8)
```

Notes:

- `time` here is **valid time**, not lead time. Lead time is
  `time - time[0]`.
- Level selection is a coordinate lookup, so `.sel(level=500)` works and
  `.sel(level=slice(1000, 500))` gives a vertical section.

## Side by side

:::{list-table}
:header-rows: 1
:widths: 26 37 37

* - Concept
  - Earth2Studio
  - ERA5-style
* - Forecast axis
  - `lead_time` (timedelta from init)
  - `time` (absolute valid time)
* - Init time
  - `time` dimension, usually length 1
  - `time[0]`
* - Vertical
  - Folded into variable name
  - `level` coordinate
* - Select 500 hPa height
  - `ds["z500"]`
  - `ds["Z"].sel(level=500)`
* - Variable naming
  - Lowercase composite (`t850`)
  - Uppercase bare (`T`)
:::

## How InferStudio detects the convention

Detection is structural, not filename-based: the dataset scanner checks for a
`lead_time` dimension and branches. Earth2Studio output takes the flattened path,
everything else is treated as ERA5-style.

This is why a hand-edited or third-party file may load in the metadata view but
present the wrong controls — if it has a `lead_time` dimension *and* a separate
level coordinate, detection will pick flattened and the level control will be
hidden.

## CF compliance

As of Earth2Studio v0.16.0, none of its IO backends (netcdf4, xarray, zarr) emit
fully CF-compliant metadata. Missing or non-conforming attributes typically
include `standard_name`, `units` on some coordinates, and the coordinate
attributes a CF-strict reader expects.

Practical consequences:

- InferStudio, xarray, and most Python tooling read the files without complaint.
- CF-strict tools — some visualization packages, THREDDS catalogues, and
  compliance checkers — will reject or mis-read them.

If you need compliance, post-process. The transformation is mechanical: attach
`standard_name` and `units` per variable from a lookup table, add `axis`
attributes to the coordinates, and set the global `Conventions` attribute. A
utility for this has been drafted; ask the maintainers for its current state
rather than writing a third copy.

## Provenance in the converted files

Each model's `<file_stem>_cf.nc`, the file InferStudio converts its output to,
says in its global attributes what was done to the data and how it was made:

| Attribute | What it holds |
|---|---|
| `title`, `institution`, `source`, `history`, `references`, `comment` | CF's own. `source` is the model and its version; `history` has the conversion, with the time and InferStudio's commit, above the input file's own history. |
| `conversion_steps` | A numbered list of every change the conversion made, in order: dimensions folded or renamed, which variables were stacked onto which levels, renames, type casts, steps added (WXFormer's empty initial step), attributes set and how the file was encoded. |
| `conversion_input`, `conversion_software`, `conversion_date`, `conversion_host` | The file(s) converted, the InferStudio commit that did it (with "+ local changes" if the checkout had any), and when and where. |
| `inference_*` | The run: `model`, `model_version` (where the weights come from, with their revision where it's pinned), `framework` and `framework_version` (earth2studio or miles-credit), `initial_conditions` or `data_source`, `initialization_time`, `forecast_length_hours`, `output_interval_hours`, `steps`, `forcing` (CAMulator), `variables`, `variables_not_written`, `run_changes` (how InferStudio's run differs from a plain one, e.g. FourCastNet3's chunked convolutions), `simulation_name`, `run_started`, `torch_version`, `gpu` and `pbs_job_id`. Only those that apply are written. |

Read them with `ncdump -h <file>` or `xarray.open_dataset(<file>).attrs`.

