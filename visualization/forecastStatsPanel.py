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

2. If the model's grid doesn't match GFS's grid exactly (e.g. Aurora's
   720-point latitude grid vs GFS's 721-point grid, which includes both
   poles), the model field is interpolated onto GFS's actual grid via
   xr.DataArray.interp_like before differencing. This assumes both
   DataArrays carry real coordinate values (not just dimension sizes) for
   lat/lon — true for standard CF-compliant output, which GFS and
   earth2studio's model outputs should both be.

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
import threading

import numpy as np
import pandas as pd
import xarray as xr
import panel as pn
import param
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dimensions import resolve_nc_glob
from visualization.earth2StudioPlot import parse_variable_groups, resolve_var_name, _resolve_dim, LAT_NAMES, LON_NAMES

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
    """Return (rmse, mae) between two 2D fields, interpolating the model
    field onto the truth field's grid first if they don't already match
    (e.g. Aurora's 720-point latitude grid vs GFS's 721-point grid)."""
    # Match dimension names first — interp_like matches by name, so if the
    # model and GFS use different names for the same axis (e.g. "lat" vs
    # "latitude"), rename the model's dims to whatever truth_da uses before
    # attempting interpolation.
    model_lat = _resolve_dim(model_da, *LAT_NAMES)
    model_lon = _resolve_dim(model_da, *LON_NAMES)
    truth_lat = _resolve_dim(truth_da, *LAT_NAMES)
    truth_lon = _resolve_dim(truth_da, *LON_NAMES)

    rename_map = {}
    if model_lat and truth_lat and model_lat != truth_lat:
        rename_map[model_lat] = truth_lat
    if model_lon and truth_lon and model_lon != truth_lon:
        rename_map[model_lon] = truth_lon
    if rename_map:
        model_da = model_da.rename(rename_map)

    if model_da.shape != truth_da.shape:
        # Grids don't line up (different resolution/point count) —
        # interpolate the model field onto the truth field's actual grid
        # rather than assuming they already match.
        model_da = model_da.interp_like(truth_da)

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

    with xr.open_mfdataset(resolve_nc_glob(model_dir), engine="netcdf4", autoclose=True, data_vars="all") as ds:
        level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))
        var_name = resolve_var_name(level_vars, surface_vars, base_or_var, level)

        lat_dim = _resolve_dim(ds, *LAT_NAMES)
        lon_dim = _resolve_dim(ds, *LON_NAMES)

        # Same lead_time-vs-time detection as scan_single_dataset in
        # app_layout.py: earth2studio forecast output has a size-1 `time`
        # (init/cycle time) plus a separate `lead_time` dim holding the
        # actual forecast steps. Iterating over `time` alone (size 1) was
        # producing only a single verification point regardless of how
        # many real forecast steps existed — prefer `lead_time` whenever
        # it's present and non-trivial.
        has_lead_time = "lead_time" in ds.sizes and ds.sizes["lead_time"] > 1

        if has_lead_time:
            select_dim = "lead_time"
            n_steps = ds.sizes["lead_time"]
            init_time = ds["time"].values[0] if "time" in ds.coords else None
            lead_time_values = ds["lead_time"].values
        else:
            select_dim = _resolve_dim(ds, "time")
            if select_dim is None:
                candidates = [d for d in ds[var_name].dims if d not in (lat_dim, lon_dim)]
                select_dim = candidates[0] if candidates else None
            if select_dim is None:
                raise ValueError(f"Could not identify a time-like dimension for {var_name!r}")
            n_steps = ds.sizes[select_dim]
            init_time = None
            lead_time_values = None

        rows = []
        for i in range(n_steps):
            model_field = ds[var_name].isel({select_dim: i})
            for extra in [d for d in model_field.dims if d not in (lat_dim, lon_dim)]:
                if model_field.sizes[extra] == 1:
                    model_field = model_field.isel({extra: 0})

            if has_lead_time and init_time is not None:
                lead_delta = lead_time_values[i]
                valid_time = init_time + lead_delta
                lead_hours = float(lead_delta / np.timedelta64(1, "h"))
            elif select_dim in ds.coords:
                valid_times = ds[select_dim].values
                valid_time = valid_times[i]
                lead_hours = float((valid_time - valid_times[0]) / np.timedelta64(1, "h"))
            else:
                valid_time = None
                lead_hours = float(i * 6)  # fallback assumption if no time coord found

            try:
                truth_field = _load_gfs_truth(valid_time, var_name)
                rmse, mae = _spatial_errors(model_field, truth_field)
                error_msg = None
            except Exception as e:
                rmse, mae = np.nan, np.nan
                error_msg = repr(e)

            rows.append({
                "lead_hours": lead_hours,
                "rmse": rmse,
                "crps": mae,
                "error": error_msg,
            })

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

        self.spinner = pn.indicators.LoadingSpinner(
            value=False, visible=False, width=20, height=20, color="primary"
        )

        self.status = pn.pane.Markdown(
            "*Click \"Compute Stats\" to verify the current variable/level "
            "against GFS analysis (fetches data over the network — may take "
            "a minute).*"
        )
        self.plot_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)

    def _on_compute_click(self, event):
        self.compute_button.disabled = True
        self.spinner.value = True
        self.spinner.visible = True
        self.status.object = "*Computing...*"

        var_name = self.controls.var_name
        level_value = self.controls.level_value

        if not var_name:
            self.status.object = "*No variable selected.*"
            self.compute_button.disabled = False
            self.spinner.value = False
            self.spinner.visible = False
            return

        # Capture the document on this (correct) callback thread before
        # handing off the slow work to a background thread — same pattern
        # used in InferenceTab, needed because setting spinner.visible=True
        # here and then blocking synchronously on slow network fetches
        # means Panel never gets a chance to flush that "show" state to
        # the browser before we'd otherwise flip it back off.
        doc = pn.state.curdoc

        def _do_compute():
            results = {}
            errors = {}
            for model in self.models:
                if model not in EARTH2STUDIO_FORMAT_MODELS:
                    continue
                try:
                    results[model] = compute_model_stats(self.model_paths[model], var_name, level_value)
                except Exception as e:
                    errors[model] = str(e)

            def _finish():
                if not results:
                    self.status.object = f"*Could not compute stats. Errors: {errors}*"
                    self.compute_button.disabled = False
                    self.spinner.value = False
                    self.spinner.visible = False
                    return

                failure_summaries = []
                for model, df in results.items():
                    failed = df[df["rmse"].isna()]
                    if not failed.empty:
                        sample_errors = failed["error"].dropna().unique()
                        sample = sample_errors[0] if len(sample_errors) else "unknown error"
                        failure_summaries.append(
                            f"{model}: {len(failed)}/{len(df)} lead times failed "
                            f"(e.g. {sample})"
                        )

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
                if failure_summaries:
                    msg += "\n\n" + "\n\n".join(failure_summaries)
                self.status.object = msg
                self.compute_button.disabled = False
                self.spinner.value = False
                self.spinner.visible = False

            if doc is not None:
                doc.add_next_tick_callback(_finish)
            else:
                _finish()

        threading.Thread(target=_do_compute, daemon=True).start()

    def panel(self):
        return pn.Column(
            pn.pane.Markdown("### Forecast Verification Statistics"),
            pn.Row(
                self.compute_button,
                self.spinner,
                align="center",
            ),
            self.status,
            self.plot_pane,
            sizing_mode="stretch_width",
            css_classes=["plot-container"],
        )
