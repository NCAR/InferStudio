"""CESM2 Large Ensemble as a physical model for the Climate tab: instead of
running a model, read one historical member's 6-hourly CAM6 output from
GDEX (d651056) for the run's dates, and write it in CAMulator's layout and
names. Everything downstream - plots, differences, video - then treats it
as one more model.

CESM2-LE's atmosphere is CAM6 on the same 1 degree grid (192x288) and 32
hybrid levels as CAMulator, so nothing is interpolated; latitudes are only
flipped to earth2studio's north-to-south order. Two things don't line up:

- CESM2-LE uses a 365-day calendar, so it has no 29 February. Steps on
  that day are written as NaN, and every other step is the same calendar
  date and hour.
- It's a free-running coupled run, so its weather is its own: compare
  statistics with a CAMulator run, not individual days.

Only the 10 members with 6-hourly 3D fields (the smoothed-biomass-burning
ones, 1011.001 to 1191.010) are offered. The read needs no GPU: pbsJob runs
it in a CPU lane of the job, or in a CPU-only job when no AI model is
selected.
"""

import os
from datetime import datetime

from inference.modelRunner import ModelRunner
from inference.earth2StudioRunner import (
    MODEL_ENV_MAP, MODEL_VAR_MAP, APP_DIR, CESM2LE_END, climate_start_times, step_hours)

MODEL = 'CESM2-LE'
CLIMATE_PHYSICAL_MODEL_LIST = [MODEL]

CESM2LE_DIR = '/gdex/data/d651056/CESM2-LE/atm/proc/tseries/hour_6'
CESM2LE_CASE = 'b.e21.BHISTsmbb.f09_g17.LE2-{member}'
CESM2LE_MEMBERS = ['1011.001', '1031.002', '1051.003', '1071.004', '1091.005',
                   '1111.006', '1131.007', '1151.008', '1171.009', '1191.010']

# Output name (CAMulator's, where it has one) -> (CESM variable, its level
# or None, factor). Z500 is geopotential height in CESM and geopotential
# in earth2studio.
GRAVITY = 9.80665
OUTPUT_VARS = {
    **{f'{name}{k}k': (cesm, k, 1.0)
       for name, cesm in (('u', 'U'), ('v', 'V'), ('t', 'T'), ('qtot', 'Q'))
       for k in range(32)},
    'sp':       ('PS', None, 1.0),
    'mtuwlwrf': ('FLUT', None, 1.0),
    'z500':     ('Z500', None, GRAVITY),
    'tcwv':     ('TMQ', None, 1.0),
}

# Why a selected field isn't written, where there's more to say than that
# CESM2-LE doesn't have it.
MISSING_REASONS = {
    't2m': "t2m (CESM2-LE saved TREFHT only daily)",
}

# Any earth2studio environment will do: this only reads files.
READ_PYTHON = MODEL_ENV_MAP['Pangu']

# Steps read and written at once: 10 days.
BLOCK_STEPS = 40


class CESM2LERunner(ModelRunner):

    def validate(self, config) -> str | None:
        if not config["simulation_name"].strip():
            return "Error: Please enter a simulation name."
        start, end = _as_datetime(config["start_time"]), _as_datetime(config["end_time"])
        if start not in climate_start_times(MODEL):
            return (f"Error: CESM2-LE runs start at 00Z on the 1st of a month, "
                    f"1850-2014; got {start:%Y-%m-%d %H}Z.")
        if end > CESM2LE_END:
            return (f"Error: CESM2-LE's historical runs end at "
                    f"{CESM2LE_END:%Y-%m-%d %H}Z; this run ends {end:%Y-%m-%d %H}Z.")
        if config.get("member", CESM2LE_MEMBERS[0]) not in CESM2LE_MEMBERS:
            return f"Error: CESM2-LE has no 6-hourly member {config['member']}."
        if not _output_vars(config)[0]:
            return ("Error: CESM2-LE has none of the selected variables. See its "
                    "button's tooltip for what it has.")
        return None

    def prepare(self, config) -> dict:
        return config

    def build_cmd(self, config) -> str:
        if not os.path.exists(READ_PYTHON):
            raise RuntimeError(f"No environment for reading {MODEL}: "
                               f"expected interpreter at {READ_PYTHON}")

        start, end = _as_datetime(config["start_time"]), _as_datetime(config["end_time"])
        hours = step_hours(config["timestep"])
        n_steps = int((end - start).total_seconds() / 3600 / hours)
        member = config.get("member", CESM2LE_MEMBERS[0])
        variables, missing = _output_vars(config)
        sources = {v: OUTPUT_VARS[v] for v in variables}
        output_nc = f"{config['output_path']}/{config['file_stem']}.nc"
        script_path = os.path.join(config["output_dir"], f"{config['file_stem']}_run.py")
        missing_warning = (f"print('WARNING: {MODEL} has no {', '.join(missing)}; "
                           "not saved.', flush=True)" if missing else "")

        script = f"""
# Read CESM2-LE member {member} from {start.isoformat()} in InferStudio's
# model output layout: one file, (time=1, lead_time, lat, lon) per variable.
import os
import sys
import time

from collections import OrderedDict
from datetime import datetime, timedelta
import cftime
import netCDF4
import numpy as np
import torch
from earth2studio.io import NetCDF4Backend

init = datetime.fromisoformat("{start.isoformat()}")
leads = [timedelta(hours={hours} * i) for i in range({n_steps} + 1)]
sources = {sources!r}
variables = list(sources)
data_dir = {CESM2LE_DIR!r}
case = {CESM2LE_CASE.format(member=member)!r}
{missing_warning}

def index(cesm_var):
    \"\"\"(year, month, day, hour) -> (file path, time index) for every
    time in this member's files of cesm_var.\"\"\"
    found = {{}}
    folder = os.path.join(data_dir, cesm_var)
    for name in sorted(os.listdir(folder)):
        if name.startswith(case + ".cam.") and name.endswith(".nc"):
            path = os.path.join(folder, name)
            with netCDF4.Dataset(path) as nc:
                t = nc["time"]
                for i, d in enumerate(cftime.num2date(t[:], t.units, t.calendar)):
                    found.setdefault((d.year, d.month, d.day, d.hour), (path, i))
    if not found:
        raise RuntimeError(f"No CESM2-LE files for {{case}} in {{folder}}")
    return found

valid = [init + lead for lead in leads]
keys = [(t.year, t.month, t.day, t.hour) for t in valid]
indexes = {{cesm: index(cesm) for cesm in sorted({{s[0] for s in sources.values()}})}}
for cesm, found in indexes.items():
    absent = [t for t, k in zip(valid, keys) if k not in found
              and not (t.month == 2 and t.day == 29)]
    if absent:
        raise RuntimeError(f"CESM2-LE {{case}} has no {{cesm}} for {{absent[0]:%Y-%m-%d %HZ}}")
leap_days = sum(1 for t in valid if t.month == 2 and t.day == 29)
if leap_days:
    print(f"CESM2-LE's calendar has no 29 February: {{leap_days}} steps on it are NaN.",
          flush=True)

files = {{}}
def read(cesm, keys_block):
    \"\"\"cesm's field at each of keys_block, (n, [lev,] lat, lon), north to
    south; NaN where the date doesn't exist.\"\"\"
    found = indexes[cesm]
    out = None
    for j, k in enumerate(keys_block):
        path, i = found.get(k, next(iter(found.values())))
        if path not in files:
            files[path] = netCDF4.Dataset(path)
            files[path].set_auto_mask(False)
        if out is None:
            shape = files[path][cesm].shape[1:]
            out = np.full((len(keys_block),) + shape, np.nan, dtype=np.float32)
        if k in found:
            out[j] = files[path][cesm][i]
    return out[..., ::-1, :]

print(f"Reading CESM2-LE {{case}} from {{init:%Y-%m-%d %HZ}}: {{len(leads)}} steps, "
      f"{{len(variables)}} variables", flush=True)
first = next(iter(indexes.values()))
with netCDF4.Dataset(next(iter(first.values()))[0]) as nc:
    lat, lon = nc["lat"][::-1].astype(np.float64), nc["lon"][:].astype(np.float64)
coords = OrderedDict(
    time=np.array([np.datetime64(init, "ns")]),
    # In hours, as the AI runs write it; NetCDF4Backend has no nanosecond
    # units.
    lead_time=np.array([int(l.total_seconds() // 3600) for l in leads],
                       dtype="timedelta64[h]"),
    lat=lat,
    lon=lon,
)
io = NetCDF4Backend("{output_nc}", backend_kwargs={{'mode': 'w'}})
io.add_array(coords, variables)

for b in range(0, len(leads), {BLOCK_STEPS}):
    started = time.monotonic()
    block = slice(b, min(b + {BLOCK_STEPS}, len(leads)))
    fields = {{}}
    for cesm in indexes:
        fields[cesm] = read(cesm, keys[block])
    tensors = []
    for v in variables:
        cesm, level, factor = sources[v]
        data = fields[cesm] if level is None else fields[cesm][:, level]
        tensors.append(torch.as_tensor(np.ascontiguousarray(data * factor)[None]))
    io.write(tensors, OrderedDict(coords, lead_time=coords["lead_time"][block]), variables)
    print(f"[{{block.stop}}/{{len(leads)}}] to {{valid[block.stop - 1]:%Y-%m-%d %HZ}} "
          f"({{time.monotonic() - started:.0f}}s)", flush=True)
for nc in files.values():
    nc.close()
io.close()
print("Output written to {output_nc}", flush=True)

print("Converting to CF-compliant format...", flush=True)
try:
    sys.path.insert(0, {APP_DIR!r})
    from cf_convert import make_cf_compliant
    cf_path = make_cf_compliant("{output_nc}")
    # Its Q is vapour only, though it's called qtot to match CAMulator's.
    with netCDF4.Dataset(cf_path, "a") as nc:
        if "qtot" in nc.variables:
            nc["qtot"].long_name = "Specific humidity (CESM2-LE: vapour only)"
    print(f"CF-compliant file written: {{cf_path}}", flush=True)
except Exception as e:
    print(f"WARNING: CF conversion failed, raw output is still available: {{e}}", flush=True)
"""
        with open(script_path, 'w') as f:
            f.write(script)
        return f"{READ_PYTHON} {script_path}"


def _output_vars(config):
    """(the selected output names CESM2-LE has, labels of those it hasn't)."""
    var_map = MODEL_VAR_MAP[MODEL]
    have, missing = [], []
    for label in config["ua_vars"] + config["surface_vars"]:
        names = var_map.get(label, [label.lower()])
        have += [n for n in names if n in OUTPUT_VARS]
        missing += [MISSING_REASONS.get(n, n) for n in names if n not in OUTPUT_VARS]
    return list(dict.fromkeys(have)), missing


def _as_datetime(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value
