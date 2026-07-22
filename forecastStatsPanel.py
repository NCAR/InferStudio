"""Computes and displays RMSE / CRPS verification statistics for the
currently selected variable+level, comparing each checked model's forecast
against GFS analysis at each forecast lead time.

IMPORTANT — things that need verifying against your actual environment:

1. `_load_gfs_truth` assumes earth2studio's GFS datasource can be called as
       GFS()(valid_time, [variable_name])
   returning an xr.DataArray indexed by (time, variable, lat, lon) (or
   similar). This matches earth2studio's general DataSource calling
   convention and the module path (earth2studio.data.gfs) already seen
   working during inference, but the *exact* signature may differ in your
   installed version. If this fails, check earth2studio's actual
   DataSource.__call__ signature and adjust accordingly.

2. This assumes the model's own lat/lon grid matches GFS's grid exactly,
   so it can diff the two fields directly. If AIFS/Aurora output on a
   different grid/resolution than GFS's 0.25deg grid, this will need an
   interpolation step (e.g. via xr.DataArray.interp_like) before diffing
   — currently it does NOT do this, so a grid mismatch would produce
   garbage (or an outright shape-mismatch error).

3. For a SINGLE DETERMINISTIC forecast (no ensemble members), CRPS reduces
   mathematically to the absolute error at each point/time — it is not
   approximated as MAE here, it IS exactly MAE for a point-mass forecast.
   This means CRPS and MAE will look identical until/unless ensemble
   members become available; the panel is labeled accordingly rather than
   implying something it isn't.

Given points 1 and 2 are unverified, it's worth testing this against a
single model / single lead time first (e.g. call compute_model_stats
directly in a notebook cell) before relying on the full UI button.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import panel as pn
import param
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from earth2StudioPlot import parse_variable_groups, resolve_var_name, _resolve_dim, LAT_NAMES, LON_NAMES

# Same set used in datasetPlot.py — models whose output this verification
# pipeline (and the earth2studio GFS datasource) can currently handle.
EARTH2STUDIO_FORMAT_MODELS = {"AIFS", "Aurora", "Pangu"}


def _load_gfs_truth(valid_time, var_name):
    """Fetch a single-variable GFS analysis field at a given valid time.
    See module docstring point 1 — verify this call signature."""
    from earth2studio.data import GFS
    source = GFS()
    da = source(valid_time, [var_name])
    da = da.squeeze()
    return da


def _spatial_errors(model_da, truth_da):
    """Return (rmse, mae) between two 2D fields. Assumes matching grids —
    see module docstring point 2."""
    diff = np.asarray(model_da.values) - np.asarray(truth_da.values)
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    mae = float(np.nanmean(np.abs(diff)))
    return rmse, mae


def compute_model_stats(model_dir, base_or_var, level) -> pd.DataFrame:
    """Compute RMSE and CRPS(=MAE) at every lead time for one model's
    output, verified against GFS analysis at each corresponding valid time.

    Returns a DataFrame with columns: lead_hours, rmse, crps
    """
    model_dir = Path(model_dir)

    with xr.open_mfdataset(f"{model_dir}/*.nc", engine="netcdf4", autoclose=True, data_vars="all") as ds:
        level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))
        var_name = resolve_var_name(level_vars, surface_vars, base_or_var, level)

        lat_dim = _resolve_dim(ds, *LAT_NAMES)
        lon_dim = _resolve_dim(ds, *LON_NAMES)
        time_dim = _resolve_dim(ds, "time")
        if time_dim is None:
            # Fall back to "whatever dim isn't lat/lon" — same approach
            # used in earth2studioPlot.plot_e2s_field
            candidates = [d for d in ds[var_name].dims if d not in (lat_dim, lon_dim)]
            time_dim = candidates[0] if candidates else None

        if time_dim is None:
            raise ValueError(f"Could not identify a time-like dimension for {var_name!r}")

        n_steps = ds.sizes[time_dim]
        valid_times = ds[time_dim].values if time_dim in ds.coords else None

        rows = []
        for i in range(n_steps):
            model_field = ds[var_name].isel({time_dim: i})
            for extra in [d for d in model_field.dims if d not in (lat_dim, lon_dim)]:
                if model_field.sizes[extra] == 1:
                    model_field = model_field.isel({extra: 0})

            if valid_times is not None:
                valid_time = valid_times[i]
                lead_hours = float((valid_time - valid_times[0]) / np.timedelta64(1, "h"))
            else:
                valid_time = None
                lead_hours = float(i * 6)  # fallback assumption if no time coord found

            try:
                truth_field = _load_gfs_truth(valid_time, var_name)
                rmse, mae = _spatial_errors(model_field, truth_field)
            except Exception:
                rmse, mae = np.nan, np.nan

            rows.append({"lead_hours": lead_hours, "rmse": rmse, "crps": mae})

    return pd.DataFrame(rows)


class ForecastStatsPanel(param.Parameterized):
    """Card shown beneath the plots: RMSE / CRPS vs lead time, one line per
    checked model, for the currently selected variable+level (from the
    shared controls). Computation is manually triggered via a button
    rather than automatic on every variable/level change, since it fetches
    GFS analysis data over the network for every lead time and can be
    slow."""

    def __init__(self, controls, dataset_key, metadata, **params):
        super().__init__(**params)
        self.controls = controls
        self.dataset_key = dataset_key
        self.metadata = metadata[dataset_key]

        if "models" in self.metadata:
            self.models = sorted(self.metadata["models"].keys())
            self.model_paths = {m: self.metadata["models"][m]["path"] for m in self.models}
        else:
            self.models = [dataset_key]
            self.model_paths = {dataset_key: self.metadata["path"]}

        self.compute_button = pn.widgets.Button(
            name="Compute Stats", button_type="primary", width=150
        )
        self.compute_button.on_click(self._on_compute_click)

        self.status = pn.pane.Markdown(
            "*Click \"Compute Stats\" to verify the current variable/level "
            "against GFS analysis (fetches data over the network — may take "
            "a minute).*"
        )
        self.plot_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)

    def _on_compute_click(self, event):
        self.compute_button.disabled = True
        self.status.object = "*Computing...*"

        var_name = self.controls.var_name
        level_value = self.controls.level_value

        if not var_name:
            self.status.object = "*No variable selected.*"
            self.compute_button.disabled = False
            return

        results = {}
        errors = {}
        for model in self.models:
            if model not in EARTH2STUDIO_FORMAT_MODELS:
                continue
            try:
                results[model] = compute_model_stats(self.model_paths[model], var_name, level_value)
            except Exception as e:
                errors[model] = str(e)

        if not results:
            self.status.object = f"*Could not compute stats. Errors: {errors}*"
            self.compute_button.disabled = False
            return

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for model, df in results.items():
            axes[0].plot(df["lead_hours"] / 24, df["rmse"], marker="o", label=model)
            axes[1].plot(df["lead_hours"] / 24, df["crps"], marker="o", label=model)

        axes[0].set_title("RMSE")
        axes[0].set_xlabel("Lead Time (days)")
        axes[0].set_ylabel(var_name)
        axes[1].set_title("CRPS (= MAE for a deterministic forecast)")
        axes[1].set_xlabel("Lead Time (days)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Verification vs GFS analysis \u2014 {var_name}")
        fig.tight_layout()

        self.plot_pane.object = fig
        plt.close(fig)

        msg = f"Computed stats for: {', '.join(results.keys())}."
        if errors:
            msg += f" Failed for: {errors}"
        self.status.object = msg
        self.compute_button.disabled = False

    def panel(self):
        return pn.Column(
            pn.pane.Markdown("### Forecast Verification Statistics"),
            self.compute_button,
            self.status,
            self.plot_pane,
            sizing_mode="stretch_width",
            css_classes=["plot-container"],
        )
