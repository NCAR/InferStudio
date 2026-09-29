"""Process-wide lock that makes the app's long netCDF jobs take turns.

The HDF5 library under netCDF4 is not built thread-safe, and xarray's own
lock only serialises individual netCDF4 calls, not whole jobs. A model
difference being written while a Statistics run read the same suite from
another thread failed with "RuntimeError: NetCDF: HDF error" partway
through the write; the same difference computed on its own succeeded.

Only the long background jobs take this lock - difference computation
(modelDiff.compute_model_difference) and verification statistics
(ForecastStatsPanel). Field loads for the plot grid do not: they run on the
Bokeh server thread, and blocking that thread behind a multi-minute job
would freeze the whole app, which is exactly what moving differences off
it was for.

Reentrant so a job that already holds it can call helpers that take it too.
"""

import threading

NC_JOB_LOCK = threading.RLock()
