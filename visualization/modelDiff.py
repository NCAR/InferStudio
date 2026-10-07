"""Compute per-variable differences between two models' CF-compliant
output within the same simulation suite, writing the result to a new
NetCDF file that can be plotted the same way as any other model output
(via load_e2s_field / plot_e2s_field), just with a diverging colormap and
a symmetric value range, since these are signed difference fields.

Saved with the simulation: (A minus B) is written to A's own directory,
next to A's output (see dimensions.diff_file_path), e.g.
<suite>/Aurora/InferStudio_Aurora_minus_Pangu_<timestamp>_diff.nc. Once a
pair's file exists it is reused - in this session or any later one that
loads the suite - rather than recomputed.

The loaders are handed that file's path in place of a model directory.
resolve_nc_glob returns it as-is, and leaves *_diff.nc files out when it
reads a model directory, so Aurora's diffs never get merged into Aurora.

The output is written to the same CF conventions cf_convert.py produces
for model output, so a difference file is readable by anything that can
read a model file. That means carrying over the auxiliary coordinates
(forecast_reference_time, forecast_period) and the coordinate attributes,
neither of which survive building a Dataset from bare DataArrays.
"""

import os
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import dask
import xarray as xr
from dask.callbacks import Callback

from dimensions import resolve_nc_glob, diff_file_path
from visualization.earth2StudioVars import _resolve_dim, LAT_NAMES, LON_NAMES
from visualization.earth2StudioPlot import invalidate_dataset, load_e2s_field
from visualization.ncJobLock import NC_JOB_LOCK


# One lock per output file. The HoloViews grid can fire several panel
# callbacks concurrently off a single parameter change; without this, two
# threads can both see a missing diff file and both start the same
# multi-gigabyte computation.
_PAIR_LOCKS = defaultdict(threading.Lock)
_PAIR_LOCKS_GUARD = threading.Lock()

# Auxiliary (non-dimension) coordinates carried over from model A.
_AUX_COORDS = ("forecast_reference_time", "forecast_period")


def _lock_for(name: str) -> threading.Lock:
    with _PAIR_LOCKS_GUARD:
        return _PAIR_LOCKS[name]


def _build_encoding(ds):
    """Encoding matching what cf_convert.py writes for model output.

    Time as hours since the initialization rather than nanoseconds since
    1970, no _FillValue on coordinates (CF 2.5.1: coordinate variables
    should not have missing values), and no _FillValue on data variables
    either - a NaN in a difference field means an interpolation failed and
    should be visible as an anomaly, not silently absorbed as an expected
    fill value.
    """
    encoding = {v: {"zlib": True, "complevel": 4, "_FillValue": None}
                for v in ds.data_vars}

    ref = None
    if "forecast_reference_time" in ds.coords:
        ref = ds["forecast_reference_time"].values
    elif "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        ref = ds["time"].values.ravel()[0]

    units = None
    if ref is not None:
        units = "hours since " + pd.Timestamp(ref).strftime("%Y-%m-%d %H:%M:%S")

    for c in ds.coords:
        if units and np.issubdtype(ds[c].dtype, np.datetime64):
            encoding[c] = {
                "units": units,
                "calendar": "proleptic_gregorian",
                "dtype": "float64",
                "_FillValue": None,
            }
        else:
            encoding[c] = {"_FillValue": None}

    return encoding


class DiffCancelled(Exception):
    """Raised by compute_model_difference when its `cancel` event is set."""


class _CancelOnEvent(Callback):
    """Aborts a dask computation once `event` is set.

    Dask calls _pretask before each task, on the thread that called
    compute(). Callbacks are global while active, though, so this only
    raises for computations started on the thread that created it -
    another thread's load or Statistics run carries on.
    """

    def __init__(self, event):
        super().__init__()
        self._event = event
        self._thread = threading.get_ident()

    def _pretask(self, key, dsk, state):
        if self._event.is_set() and threading.get_ident() == self._thread:
            raise DiffCancelled()


def compute_model_difference(model_a_dir, model_b_dir,
                             model_a_name, model_b_name,
                             cancel=None) -> Path:
    """Compute (model_a - model_b) for every variable present in both
    datasets, writing the result to diff_file_path(model_a_dir, ...) -
    in model A's directory.

    Returns the path to that file, which the loaders accept wherever they
    take a model directory. If the file already exists, it is returned
    immediately without recomputing.

    `cancel`, a threading.Event, stops the computation when set: it raises
    DiffCancelled at the next dask task, leaving no file behind.
    """
    cancel = cancel or threading.Event()
    out_path = diff_file_path(model_a_dir, model_a_name, model_b_name)

    # Fast path outside the lock - the overwhelmingly common case once a
    # pair has been computed is that the file is simply there.
    if out_path.exists():
        return out_path

    with _lock_for(str(out_path)):
        # Re-check: another thread may have finished while we waited.
        if out_path.exists():
            return out_path

        # Write to a temporary name in the same directory, then rename.
        # os.replace is atomic within a filesystem, so a reader either sees
        # no file or a complete one - never a partially flushed NetCDF.
        # This matters now that plot callbacks poll these paths on every
        # parameter change rather than once per explicit user action.
        # The leading dot and .tmp also keep it out of the *.nc glob.
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")

        # NC_JOB_LOCK first, so a Statistics run reading the suite waits
        # for this whole open/compute/write rather than interleaving HDF5
        # calls with it - see ncJobLock.py.
        with NC_JOB_LOCK, \
             xr.open_mfdataset(resolve_nc_glob(model_a_dir), engine="netcdf4",
                               data_vars="all", chunks={}) as ds_a, \
             xr.open_mfdataset(resolve_nc_glob(model_b_dir), engine="netcdf4",
                               data_vars="all", chunks={}) as ds_b:

            # Waiting for NC_JOB_LOCK can take a while if Statistics holds it.
            if cancel.is_set():
                raise DiffCancelled()

            shared_vars = sorted(set(ds_a.data_vars) & set(ds_b.data_vars))
            if not shared_vars:
                raise ValueError(
                    f"{model_a_name} and {model_b_name} have no variables in "
                    f"common \u2014 cannot compute a difference."
                )

            diff_vars = {}
            skipped = {}
            for var in shared_vars:
                da_a = ds_a[var]
                da_b = ds_b[var]
                try:
                    if da_a.shape != da_b.shape:
                        # Grids don't line up (e.g. AIFS's 721-point latitude
                        # grid vs Aurora's 720-point grid). Interpolate ONLY
                        # the specific dimensions that actually mismatch, by
                        # name - NOT a blanket .interp_like(), which tries to
                        # interpolate every matching coordinate (including
                        # size-1 dims like "time"). Scipy's linear
                        # interpolator needs >=2 points to compute a slope; a
                        # size-1 axis hits an exact 0/0 division, and that
                        # single NaN then propagates through the ENTIRE array
                        # via broadcasting - silently turning a minor,
                        # legitimate lat-grid mismatch into 100% NaN output.
                        #
                        # The target grid is picked by NAME (alphabetically
                        # first of model_a_name/model_b_name), not by
                        # position (always "a"). Picking by position meant
                        # compute_model_difference(Aurora, Pangu) regridded
                        # onto Aurora's grid while the (Pangu, Aurora) call
                        # regridded onto Pangu's grid - two independently
                        # interpolated fields that only approximately
                        # mirrored each other. Picking by name is the same
                        # regardless of which one is passed as "a", so both
                        # directions land on the identical target grid and
                        # diff(A, B) == -diff(B, A) exactly, not just
                        # approximately.
                        a_is_target = model_a_name <= model_b_name
                        target_da = da_a if a_is_target else da_b

                        lat_dim = _resolve_dim(da_a, *LAT_NAMES)
                        lon_dim = _resolve_dim(da_a, *LON_NAMES)
                        interp_kwargs = {}
                        for dim in (lat_dim, lon_dim):
                            if (
                                dim
                                and dim in da_b.dims
                                and dim in da_a.dims
                                and (
                                    da_a.sizes[dim] != da_b.sizes[dim]
                                    or not np.array_equal(
                                        da_a[dim].values, da_b[dim].values)
                                )
                            ):
                                interp_kwargs[dim] = target_da[dim]
                        if interp_kwargs:
                            if a_is_target:
                                da_b = da_b.interp(**interp_kwargs)
                            else:
                                da_a = da_a.interp(**interp_kwargs)
                    diff = da_a - da_b
                    diff.attrs = dict(da_a.attrs)
                    diff_vars[var] = diff
                except Exception as e:
                    skipped[var] = str(e)

            if not diff_vars:
                raise ValueError(
                    f"Could not compute a difference for any shared variable "
                    f"between {model_a_name} and {model_b_name}: {skipped}"
                )

            out_ds = xr.Dataset(diff_vars)

            # Carry over the auxiliary coordinates from model A. A Dataset
            # built from DataArrays inherits each array's own DIMENSION
            # coordinates but not dataset-level auxiliary ones, so without
            # this the difference files come out structurally poorer than
            # the model files they came from: no forecast_period, no
            # forecast_reference_time.
            for coord in _AUX_COORDS:
                if coord in ds_a.coords:
                    out_ds = out_ds.assign_coords({coord: ds_a[coord]})
                    out_ds[coord].attrs = dict(ds_a[coord].attrs)

            # Dimension coordinates survive the DataArray round-trip as
            # values but lose their attributes, so restore those too -
            # otherwise the CF metadata cf_convert took care to write is
            # silently dropped from every difference file.
            for coord in out_ds.coords:
                if coord in ds_a.coords and not out_ds[coord].attrs:
                    out_ds[coord].attrs = dict(ds_a[coord].attrs)

            # Variable-level attributes were copied from model A above, but
            # some of them are actively wrong for a difference field. The
            # units carry over correctly (a difference of two fields in
            # K is in K), but standard_name does not: the difference of two
            # eastward_wind fields is not itself an eastward_wind, and a
            # reader trusting that name would apply the wrong valid range
            # and sign conventions.
            for v in out_ds.data_vars:
                attrs = dict(out_ds[v].attrs)
                attrs.pop("standard_name", None)
                base_long = attrs.get("long_name", v)
                attrs["long_name"] = (
                    f"{base_long} difference "
                    f"({model_a_name} minus {model_b_name})")
                if _AUX_COORDS and any(c in out_ds.coords for c in _AUX_COORDS):
                    attrs["coordinates"] = " ".join(
                        c for c in _AUX_COORDS if c in out_ds.coords)
                out_ds[v].attrs = attrs

            out_ds.attrs["Conventions"] = ds_a.attrs.get("Conventions", "CF-1.11")
            out_ds.attrs["history"] = (
                f"Difference: {model_a_name} minus {model_b_name}")
            if skipped:
                out_ds.attrs["skipped_variables"] = ", ".join(sorted(skipped))

            try:
                # Single-threaded on purpose. On dask's threaded scheduler
                # the reads of A and B and the write of the result run at
                # once, and they deadlock inside xarray: reads take
                # CombinedLock({netCDF-C, HDF5}), the write takes
                # CombinedLock({netCDF-C, HDF5, file}), and CombinedLock
                # orders its locks by set iteration - by object id, so the
                # two can take the shared pair in opposite orders. Whether
                # a process hangs depends on where those locks landed in
                # memory. Little is lost: every read and write already
                # queues for the HDF5 lock.
                with dask.config.set(scheduler="synchronous"), \
                     _CancelOnEvent(cancel):
                    out_ds.to_netcdf(tmp_path, encoding=_build_encoding(out_ds))
                os.replace(tmp_path, out_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

        # A failed or superseded earlier read of this file may be sitting
        # in the loader's open-dataset cache. Drop it so the next read
        # picks up the file just written.
        invalidate_dataset(out_path)

    return out_path


def load_diff_field(model_a_dir, model_b_dir,
                    model_a_name, model_b_name, base_or_var, level, t):
    """Convenience wrapper: ensure the diff exists, then load one field
    from it. This is what the plot grid's diff_provider calls."""
    diff_path = compute_model_difference(
        model_a_dir, model_b_dir, model_a_name, model_b_name)
    return load_e2s_field(diff_path, base_or_var, level, t)


def symmetric_diff_range(model_a_dir, model_b_dir,
                         model_a_name, model_b_name, base_or_var, level,
                         sample_steps=3):
    """Symmetric (-m, +m) colour limits sampled across a few steps.

    Symmetry is not cosmetic: with coolwarm and an asymmetric range, zero
    lands somewhere off-white and an unbiased difference field reads as
    biased.
    """
    from visualization.earth2StudioPlot import field_range
    diff_path = compute_model_difference(
        model_a_dir, model_b_dir, model_a_name, model_b_name)
    lo, hi = field_range(diff_path, base_or_var, level, sample_steps)
    m = max(abs(lo), abs(hi))
    return (-m, m)
