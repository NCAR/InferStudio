"""Physical (NWP) models: instead of running a model, fetch the operational
forecast it already published, from the same initialization and for the
same lead times and variables as the AI models, and write it into the suite
in the same layout. Everything downstream - plots, differences, video,
statistics - then treats it as one more model.

Currently GFS, from NOAA's archive on AWS through earth2studio's GFS_FX.
Another earth2studio forecast source (IFS_FX, GEFS_FX, ...) is a new entry
in FORECAST_SOURCES plus its checks in validate().

The fetch needs no GPU: pbsJob runs it in a CPU lane of the job, or in a
CPU-only job when no AI model is selected.
"""

import os
from datetime import datetime, timedelta

from inference.modelRunner import ModelRunner
from inference.earth2StudioRunner import (
    MODEL_ENV_MAP, MODEL_VAR_MAP, APP_DIR, step_hours)

# model name -> earth2studio.data forecast source class
FORECAST_SOURCES = {
    'GFS': 'GFS_FX',
}
PHYSICAL_MODEL_LIST = list(FORECAST_SOURCES)
PHYSICAL_MODELS = frozenset(PHYSICAL_MODEL_LIST)

# Any earth2studio environment will do: this only downloads.
FETCH_PYTHON = MODEL_ENV_MAP['Pangu']

# NOAA's AWS archive starts with the 2021-02-27 00Z cycle. GFS's 0.25 degree
# output is hourly to 120 h, then 3-hourly to 384 h.
GFS_FIRST_CYCLE = datetime(2021, 2, 27)
GFS_HOURLY_UNTIL = 120
GFS_MAX_LEAD = 384
# A cycle's files appear over the few hours after its nominal time.
GFS_PUBLISH_DELAY = timedelta(hours=6)

# How often a lead time's fetch is tried before the run gives up.
FETCH_ATTEMPTS = 3


class ForecastSourceRunner(ModelRunner):

    def validate(self, config) -> str | None:
        if not config["simulation_name"].strip():
            return "Error: Please enter a simulation name."
        model = config["model"]
        if model not in FORECAST_SOURCES:
            return f"Error: '{model}' is not a recognized physical model."
        start, end = _as_datetime(config["start_time"]), _as_datetime(config["end_time"])
        hours = step_hours(config["timestep"])
        lead = (end - start).total_seconds() / 3600
        if start < GFS_FIRST_CYCLE:
            return (f"Error: GFS forecasts are archived from "
                    f"{GFS_FIRST_CYCLE:%Y-%m-%d %H}Z on; this run starts "
                    f"{start:%Y-%m-%d %H}Z.")
        if start > datetime.utcnow() - GFS_PUBLISH_DELAY:
            return (f"Error: the {start:%Y-%m-%d %H}Z GFS forecast isn't "
                    "published yet. Pick an earlier start.")
        if start.hour % 6 or start.minute:
            return "Error: GFS runs at 00, 06, 12 and 18Z only."
        if lead > GFS_MAX_LEAD:
            return (f"Error: GFS forecasts go out to {GFS_MAX_LEAD} h; this run "
                    f"is {lead:g} h long.")
        if lead > GFS_HOURLY_UNTIL and hours % 3:
            return (f"Error: GFS output is 3-hourly after {GFS_HOURLY_UNTIL} h, "
                    f"so a {lead:g} h run needs an output frequency of 3 h or "
                    "a multiple of it.")
        return None

    def prepare(self, config) -> dict:
        return config

    def build_cmd(self, config) -> str:
        model = config["model"]
        if not os.path.exists(FETCH_PYTHON):
            raise RuntimeError(f"No environment for fetching {model}: "
                               f"expected interpreter at {FETCH_PYTHON}")

        start, end = _as_datetime(config["start_time"]), _as_datetime(config["end_time"])
        hours = step_hours(config["timestep"])
        n_steps = int((end - start).total_seconds() / 3600 / hours)
        variables = _translate_vars(config["ua_vars"] + config["surface_vars"], model)
        output_nc = f"{config['output_path']}/{config['file_stem']}.nc"
        script_path = os.path.join(config["output_dir"], f"{config['file_stem']}_run.py")
        source = FORECAST_SOURCES[model]

        script = f"""
# Fetch the {model} forecast from {start.isoformat()} in InferStudio's model
# output layout: one file, (time=1, lead_time, lat, lon) per variable.
import os
import sys
import time
os.environ["TQDM_DISABLE"] = "1"

from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np
import torch
from loguru import logger
from earth2studio.data import {source}
from earth2studio.io import NetCDF4Backend

init = datetime.fromisoformat("{start.isoformat()}")
leads = [timedelta(hours={hours} * i) for i in range({n_steps} + 1)]
variables = {variables!r}

# earth2studio logs every byte-range request at DEBUG.
logger.remove()
logger.add(sys.stderr, level="WARNING")

# No cache: files are only read once, and a cut-off download would stay
# behind in the cache as a truncated file.
source = {source}(cache=False, verbose=False)
io = NetCDF4Backend("{output_nc}", backend_kwargs={{'mode': 'w'}})

def fetch(lead):
    for attempt in range(1, {FETCH_ATTEMPTS} + 1):
        try:
            return source(init, lead, variables)
        except Exception as e:
            print(f"WARNING: fetching +{{lead}} failed (attempt {{attempt}}): {{e!r}}", flush=True)
            if attempt == {FETCH_ATTEMPTS}:
                raise
            time.sleep(30 * attempt)

print(f"Fetching {model} from {{init:%Y-%m-%d %HZ}}: {{len(leads)}} lead times, "
      f"{{len(variables)}} variables", flush=True)
for i, lead in enumerate(leads):
    started = time.monotonic()
    da = fetch(lead)
    if i == 0:
        coords = OrderedDict(
            time=np.array([np.datetime64(init, "ns")]),
            # In hours, as the AI runs write it; NetCDF4Backend has no
            # nanosecond units.
            lead_time=np.array([int(l.total_seconds() // 3600) for l in leads],
                               dtype="timedelta64[h]"),
            lat=da["lat"].values,
            lon=da["lon"].values,
        )
        io.add_array(coords, variables)
    step = OrderedDict(coords, lead_time=coords["lead_time"][i:i + 1])
    io.write([torch.as_tensor(da.sel(variable=v).values) for v in variables], step, variables)
    print(f"[{{i + 1}}/{{len(leads)}}] +{{int(lead.total_seconds() // 3600)}} h "
          f"({{time.monotonic() - started:.0f}}s)", flush=True)
io.close()
print("Output written to {output_nc}", flush=True)

print("Converting to CF-compliant format...", flush=True)
try:
    sys.path.insert(0, {APP_DIR!r})
    from cf_convert import make_cf_compliant
    cf_path = make_cf_compliant("{output_nc}")
    print(f"CF-compliant file written: {{cf_path}}", flush=True)
except Exception as e:
    print(f"WARNING: CF conversion failed, raw output is still available: {{e}}", flush=True)
"""
        with open(script_path, 'w') as f:
            f.write(script)
        return f"{FETCH_PYTHON} {script_path}"


def _as_datetime(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _translate_vars(names, model):
    var_map = MODEL_VAR_MAP.get(model, {})
    result = []
    for v in names:
        result.extend(var_map.get(v, [v.lower()]))
    return list(dict.fromkeys(result))
