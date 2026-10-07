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

from datetime import datetime
from pathlib import Path
import json
import os
import threading
import weakref

import numpy as np
import pandas as pd
import xarray as xr
import panel as pn
import param
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dimensions import resolve_nc_glob
from visualization.ncJobLock import NC_JOB_LOCK
from visualization.earth2StudioVars import parse_variable_groups, resolve_var_name, _resolve_dim, LAT_NAMES, LON_NAMES

# Same set used in datasetPlot.py — models whose output this verification
# pipeline (and the earth2studio GFS datasource) can currently handle.
EARTH2STUDIO_FORMAT_MODELS = {"AIFS", "Aurora", "Pangu"}

# Computed stats are saved inside each model's directory, at
# <suite>/<Model>/stats/<var>[_<level>].csv plus a .json sidecar, so a new
# session can show them without refetching GFS. A subdirectory is invisible
# to resolve_nc_glob (top-level *.nc only) and to the suite checks (which
# only look at the suite root). The sidecar records what the numbers were
# computed from; if any of it no longer matches, the saved stats are ignored.
STATS_DIRNAME = "stats"
STATS_VERSION = 1           # bump when the error computation changes
TRUTH_SOURCE = "GFS analysis"


def _stats_paths(model_dir, var_name, level):
    stem = f"{var_name}_{level}" if level else var_name
    stats_dir = Path(model_dir) / STATS_DIRNAME
    return stats_dir / f"{stem}.csv", stats_dir / f"{stem}.json"


def _source_fingerprint(model_dir):
    """Name, size and mtime of each output file the stats are computed
    from - changes if the model is re-run into the same directory."""
    fingerprint = []
    for f in resolve_nc_glob(model_dir):
        st = os.stat(f)
        fingerprint.append({"name": Path(f).name, "size": st.st_size,
                            "mtime_ns": st.st_mtime_ns})
    return fingerprint


def load_saved_stats(model_dir, var_name, level):
    """(DataFrame, computed_at) for saved stats that are still current,
    or None if there are none or they are stale."""
    csv_path, meta_path = _stats_paths(model_dir, var_name, level)
    if not (csv_path.exists() and meta_path.exists()):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if (meta.get("version") != STATS_VERSION
            or meta.get("truth") != TRUTH_SOURCE
            or meta.get("sources") != _source_fingerprint(model_dir)):
        return None
    df = pd.read_csv(csv_path)
    # A blank CSV cell reads back as NaN; keep "no error" as None.
    df["error"] = df["error"].astype(object).where(df["error"].notna(), None)
    return df, meta.get("computed_at")


def _write_atomically(path, write):
    """Write via a hidden temp file and rename, so a concurrent reader or an
    interrupted write never sees a partial file."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    write(tmp_path)
    os.replace(tmp_path, path)


def save_stats(model_dir, var_name, level, df):
    """Save one model's stats; returns the computed_at timestamp. Raises
    OSError if the suite isn't writable."""
    csv_path, meta_path = _stats_paths(model_dir, var_name, level)
    csv_path.parent.mkdir(exist_ok=True)
    computed_at = datetime.now().isoformat(timespec="minutes")
    meta = {
        "version": STATS_VERSION,
        "truth": TRUTH_SOURCE,
        "variable": var_name,
        "level": level,
        "computed_at": computed_at,
        "sources": _source_fingerprint(model_dir),
    }
    _write_atomically(csv_path, lambda p: df.to_csv(p, index=False))
    # Sidecar last: stats only count as saved once it exists.
    def _write_meta(p):
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)
    _write_atomically(meta_path, _write_meta)
    return computed_at


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


def compute_model_stats(model_dir, base_or_var, level, reuse=None) -> pd.DataFrame:
    """Compute RMSE and CRPS(=MAE) at every lead time for one model's
    output, verified against GFS analysis at each corresponding valid time.

    `reuse` is an earlier result for the same model and variable; lead
    times that succeeded there are copied rather than refetched, so only
    missing or failed ones are computed.

    Returns a DataFrame with columns: lead_hours, rmse, crps, error
    """
    model_dir = Path(model_dir)
    done = {}
    if reuse is not None:
        for row in reuse[reuse["rmse"].notna()].to_dict("records"):
            done[float(row["lead_hours"])] = row

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

            if lead_hours in done:
                prev = done[lead_hours]
                rows.append({"lead_hours": lead_hours, "rmse": prev["rmse"],
                             "crps": prev["crps"], "error": None})
                continue

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
    slow. Results are saved in the suite (see save_stats), and saved
    results for the selected variable/level are shown as soon as it is
    selected."""

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
        self.stats_models = [m for m in self.models if m in EARTH2STUDIO_FORMAT_MODELS]

        self.compute_button = pn.widgets.Button(
            name="Compute Stats", button_type="primary", width=150
        )
        self.compute_button.on_click(self._on_compute_click)

        self.spinner = pn.indicators.LoadingSpinner(
            value=False, visible=False, width=20, height=20, color="primary"
        )

        self.status = pn.pane.Markdown("")
        self.plot_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)

        # {model: (DataFrame, computed_at)} saved in the suite for the
        # selected variable/level - see _load_saved.
        self.saved = {}
        self.results = None
        self.last_var_name = None
        self.last_level = None

        # A new panel is built whenever the shown suite changes, and nothing
        # tells the old one, so watch through a weak reference: once this
        # panel is gone the watcher removes itself instead of keeping it
        # alive.
        on_change = weakref.WeakMethod(self._on_selection_change)
        def _watch(event):
            callback = on_change()
            if callback is None:
                try:
                    controls.param.unwatch(watcher)
                except Exception:
                    pass
                return
            callback(event)
        watcher = controls.param.watch(_watch, ["var_name", "level_value"])

        self._load_saved()

    def _on_selection_change(self, event):
        # A compute in progress shows its own status and calls _load_saved
        # when it finishes.
        if not self.compute_button.disabled:
            self._load_saved()

    def _load_saved(self):
        """Show the saved stats for the selected variable/level, if any."""
        var_name = self.controls.var_name
        level_value = self.controls.level_value
        self.saved = {}
        for model in self.stats_models:
            try:
                saved = load_saved_stats(self.model_paths[model], var_name, level_value)
            except Exception:
                saved = None    # unreadable or half-written: treat as missing
            if saved is not None:
                self.saved[model] = saved

        self.results = {m: df for m, (df, _) in self.saved.items()} or None
        self.last_var_name = var_name if self.saved else None
        self.last_level = level_value if self.saved else None

        if not self.saved:
            self.plot_pane.object = None
            self.status.object = (
                "*Click \"Compute Stats\" to verify the current variable/level "
                "against GFS analysis (fetches data over the network — may take "
                "a minute). Results are saved with the suite.*")
        else:
            self._render(self.results, var_name)
            dates = sorted({at for _, at in self.saved.values() if at})
            msg = (f"Loaded saved stats for: {', '.join(self.saved)}"
                   + (f" (computed {', '.join(d.replace('T', ' ') for d in dates)})." if dates else "."))
            missing = [m for m in self.stats_models if m not in self.saved]
            if missing:
                msg += f" Not yet computed for: {', '.join(missing)}."
            msg += "".join("\n\n" + f for f in self._failure_summaries(self.results))
            self.status.object = msg
        self._update_button()

    def _incomplete_models(self):
        """Models whose saved stats are missing or have failed lead times."""
        return [m for m in self.stats_models
                if m not in self.saved or self.saved[m][0]["rmse"].isna().any()]

    def _update_button(self):
        if not self.saved:
            self.compute_button.name = "Compute Stats"
        elif self._incomplete_models():
            self.compute_button.name = "Compute Missing"
        else:
            self.compute_button.name = "Recompute"

    @staticmethod
    def _failure_summaries(results):
        summaries = []
        for model, df in results.items():
            failed = df[df["rmse"].isna()]
            if not failed.empty:
                sample_errors = failed["error"].dropna().unique()
                sample = sample_errors[0] if len(sample_errors) else "unknown error"
                summaries.append(
                    f"{model}: {len(failed)}/{len(df)} lead times failed "
                    f"(e.g. {sample})"
                )
        return summaries

    def _render(self, results, var_name):
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
        fig.suptitle(f"Verification vs GFS analysis — {var_name}")
        fig.tight_layout()

        self.plot_pane.object = fig
        plt.close(fig)

    def _on_compute_click(self, event):
        var_name = self.controls.var_name
        level_value = self.controls.level_value

        if not var_name:
            self.status.object = "*No variable selected.*"
            return

        self.compute_button.disabled = True
        self.spinner.value = True
        self.spinner.visible = True
        self.status.object = "*Computing...*"

        # "Recompute" (everything already saved) starts from scratch;
        # otherwise saved lead times are reused and only the gaps fetched.
        recompute = self.compute_button.name == "Recompute"
        prior = {} if recompute else {m: df for m, (df, _) in self.saved.items()}

        # Capture the document on this (correct) callback thread before
        # handing off the slow work to a background thread — same pattern
        # used in InferenceTab, needed because setting spinner.visible=True
        # here and then blocking synchronously on slow network fetches
        # means Panel never gets a chance to flush that "show" state to
        # the browser before we'd otherwise flip it back off.
        doc = pn.state.curdoc

        def _set_status(text):
            def apply():
                self.status.object = text
            if doc is not None:
                doc.add_next_tick_callback(apply)
            else:
                apply()

        def _do_compute():
            results = {}
            errors = {}
            save_errors = {}
            # Take turns with difference computation - see ncJobLock.py.
            # Non-blocking first, only to tell the user why nothing is
            # happening yet; a difference can hold this for minutes.
            if not NC_JOB_LOCK.acquire(blocking=False):
                _set_status("*Waiting for a difference computation on the "
                            "Visualization tab to finish before starting...*")
                NC_JOB_LOCK.acquire()
                _set_status("*Computing...*")
            try:
                for model in self.stats_models:
                    reuse = prior.get(model)
                    if reuse is not None and reuse["rmse"].notna().all():
                        results[model] = reuse      # already complete
                        continue
                    try:
                        df = compute_model_stats(self.model_paths[model], var_name,
                                                 level_value, reuse=reuse)
                    except Exception as e:
                        errors[model] = str(e)
                        continue
                    results[model] = df
                    # Save each model as it finishes, so an interrupted
                    # run keeps what it got.
                    try:
                        save_stats(self.model_paths[model], var_name, level_value, df)
                    except OSError as e:
                        save_errors[model] = str(e)
            finally:
                NC_JOB_LOCK.release()

            def _finish():
                self.compute_button.disabled = False
                self.spinner.value = False
                self.spinner.visible = False

                # Selection changed while computing: what was computed is
                # saved; show whatever is saved for the new selection.
                if (var_name, level_value) != (self.controls.var_name,
                                               self.controls.level_value):
                    self._load_saved()
                    return

                if not results:
                    self.status.object = f"*Could not compute stats. Errors: {errors}*"
                    self._update_button()
                    return

                self._render(results, var_name)
                self.results = results
                self.last_var_name = var_name
                self.last_level = level_value

                # Refresh saved state (and so the button label) from disk.
                self.saved = {}
                for model in results:
                    try:
                        saved = load_saved_stats(self.model_paths[model], var_name, level_value)
                    except Exception:
                        saved = None
                    if saved is not None:
                        self.saved[model] = saved
                self._update_button()

                msg = f"Computed stats for: {', '.join(results.keys())}."
                if errors:
                    msg += f" Failed for: {errors}"
                if save_errors:
                    msg += (f"\n\nCould not save stats for "
                            f"{', '.join(save_errors)} (is the suite writable?) - "
                            f"they will need recomputing next session.")
                msg += "".join("\n\n" + f for f in self._failure_summaries(results))
                self.status.object = msg

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
