"""
Initial conditions for earth2studio's CAMulator.

CAMulator emulates CAM6 on its own 32 hybrid-sigma levels, so no analysis
source (GFS, ARCO, IFS) can start it. Its HuggingFace repo instead ships CAM
states for 1 January and 1 July 00Z of 1980-2014 (1 January only in 1980),
as CREDIT input tensors. earth2studio doesn't wrap these; this does.

Each file is the network's full normalized input, (1, 136, 1, 192, 288):
the 130 prognostic channels in earth2studio's PROGNOSTIC_VARIABLES order
(u, v, t, qtot on levels 0-31, then sp and t2m), then 2 static and 4 forcing
channels, with latitude running south to north. Only the prognostic
channels are kept. They are denormalized the way CAMulator._normalize_state
normalizes them, inverted (sp in float64, as there), and flipped to
earth2studio's north-to-south grid, so the model reproduces the CREDIT state
exactly.

    from camulatorInit import CAMulatorInitialConditions
    data = CAMulatorInitialConditions()
    da = data([datetime(2000, 1, 1)], ["t31k", "sp"])
"""

from datetime import datetime

import numpy as np
import torch
import xarray as xr

from earth2studio.data.camulator import (
    CAMULATOR_GRID_LAT,
    CAMULATOR_GRID_LON,
)
from earth2studio.models.px.camulator import (
    _MEAN_FILE,
    _STD_FILE,
    PROGNOSTIC_VARIABLES,
    CAMulator,
    _stats_field,
)

IC_FILE = "initial_conditions/init_camulator_condition_tensor_{:%Y-%m-%dT%H}Z.pth"


def available_times():
    """Start times with a shipped initial condition."""
    return [datetime(y, m, 1) for y in range(1980, 2015) for m in (1, 7)
            if (y, m) != (1980, 7)]


class CAMulatorInitialConditions:
    """earth2studio data source serving CAMulator's shipped initial states.

    Files come from the same pinned package (and so the same cache) as the
    model's weights.
    """

    def __init__(self):
        self.package = CAMulator.load_default_package()
        self._center = self._scale = None

    def __call__(self, time, variable):
        times = [time] if not isinstance(time, (list, tuple, np.ndarray)) else list(time)
        times = [_as_datetime(t) for t in times]
        variables = [variable] if isinstance(variable, str) else list(variable)

        unknown = [v for v in variables if v not in PROGNOSTIC_VARIABLES]
        if unknown:
            raise KeyError(f"CAMulator initial conditions have no {unknown}")
        idx = [PROGNOSTIC_VARIABLES.index(v) for v in variables]

        data = np.stack([self._state(t)[idx] for t in times])
        return xr.DataArray(
            data.astype(np.float32),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": np.array(times, dtype="datetime64[ns]"),
                "variable": np.array(variables),
                "lat": CAMULATOR_GRID_LAT,
                "lon": CAMULATOR_GRID_LON,
            },
        )

    def _state(self, time):
        """Physical prognostic state at `time`, (130, lat, lon), float64."""
        if time not in available_times():
            raise ValueError(
                f"CAMulator has no initial condition for {time:%Y-%m-%d %H}Z; "
                "it ships states for 1 January and 1 July 00Z of 1980-2014 "
                "(1 January only in 1980)")
        path = self.package.resolve(IC_FILE.format(time))
        x = torch.load(path, map_location="cpu", weights_only=True)
        x = x[0, :len(PROGNOSTIC_VARIABLES), 0].numpy().astype(np.float64)
        center, scale = self._stats()
        return (x * scale + center)[:, ::-1]

    def _stats(self):
        if self._center is None:
            shape = (len(CAMULATOR_GRID_LAT), len(CAMULATOR_GRID_LON))
            mean = xr.open_dataset(self.package.resolve(_MEAN_FILE)).load()
            std = xr.open_dataset(self.package.resolve(_STD_FILE)).load()
            self._center = np.stack(
                [_stats_field(mean, v, shape) for v in PROGNOSTIC_VARIABLES])
            self._scale = np.stack(
                [_stats_field(std, v, shape) for v in PROGNOSTIC_VARIABLES])
        return self._center, self._scale


def _as_datetime(t):
    if isinstance(t, datetime):
        return t
    return datetime.fromisoformat(str(np.datetime_as_string(np.datetime64(t, "s"))))

