from inference.modelRunner import ModelRunner
from datetime import datetime, timedelta
import os

# This checkout (SandboxInferStudio or ProductionInferStudio), where the
# job scripts import cf_convert from.
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step_hours(timestep):
    """Hours in a TimePicker timestep such as "6h"."""
    return int(timestep.replace("h", ""))

#EARTH2STUDIO_PYTHON = f"/glade/work/{os.environ['USER']}/conda-envs/earth2studio/bin/python"
#EARTH2STUDIO_PYTHON = f"/glade/work/pearse/conda-envs/earth2studio/bin/python"
MODEL_ENV_MAP = {
    'AIFS':         f"/glade/work/pearse/E2S/envs/aifs/bin/python",
    'CAMulator':    f"/glade/work/pearse/E2S/envs/camulator/bin/python",
    'DLESyMv0_ISCCP_ERA5': f"/glade/work/pearse/E2S/envs/dlesym/bin/python",
    'Aurora':       f"/glade/work/pearse/E2S/envs/aurora/bin/python",
    'FourCastNet3': f"/glade/work/pearse/E2S/envs/fcn3/bin/python",
    'GraphCast':    f"/glade/work/pearse/E2S/envs/graphcast/bin/python",
    'Pangu':        f"/glade/work/pearse/E2S/envs/pangu/bin/python",
    'SFNO':         f"/glade/work/pearse/E2S/envs/sfno/bin/python",
}

MODEL_MAP = {
    'AIFS':         'earth2studio.models.px.AIFS',
    'CAMulator':    'earth2studio.models.px.CAMulator',
    # The lat/lon wrapper takes ERA5 on its 0.25 degree grid and regrids
    # to and from the model's HEALPix grid itself.
    'DLESyMv0_ISCCP_ERA5': 'earth2studio.models.px.DLESyMv0_ISCCP_ERA5LatLon',
    'Aurora':       'earth2studio.models.px.Aurora',
    'FourCastNet3': 'earth2studio.models.px.FCN3',      # was FourCastNetv2Small — wrong model entirely
    'GraphCast':    'earth2studio.models.px.GraphCastSmall',
    'Pangu':        'earth2studio.models.px.Pangu6',
    'SFNO':         'earth2studio.models.px.SFNO',
}

# Initial-condition source per model. GFS lacks fields some models need
# (e.g. AIFS's skt, tcw, stl1/2, swvl1/2), and earth2studio silently fills
# missing variables with zeros, so those models blow up when started from GFS.
# Each entry is a list of (earliest start time, source class) — the latest
# entry whose start time is <= the run's start wins.
DEFAULT_DATA_SOURCE = 'GFS'
MODEL_DATA_SOURCE = {
    'AIFS': [
        (datetime.min,         'ARCO'),   # ERA5 reanalysis, for historical starts
        (datetime(2024, 3, 1), 'IFS'),    # ECMWF open data (earliest IFS archive date)
    ],
    # CAMulator runs on CAM6's own hybrid levels, which no analysis source
    # supplies; it starts from the CAM states shipped with its weights.
    'CAMulator': [
        (datetime.min,         'CAMulatorInitialConditions'),
    ],
    'DLESyMv0_ISCCP_ERA5': [
        (datetime.min,         'ARCO'),
    ],
}

# Sources that live in this directory rather than in earth2studio.data,
# by class name -> module.
LOCAL_DATA_SOURCES = {
    'CAMulatorInitialConditions': 'camulatorInit',
}

# The start times CAMulatorInitialConditions has a state for: 00Z on
# 1 January and 1 July, 1980-2014, except July 1980. Kept here as well as in
# camulatorInit, which imports earth2studio and so can't be imported by the
# app.
CAMULATOR_START_TIMES = frozenset(
    datetime(y, m, 1) for y in range(1980, 2015) for m in (1, 7)
    if (y, m) != (1980, 7))

# CAMulatorForcing's transient record covers 1980-2014. A step reads the
# forcing at its input time, so a run can end 6 hours after the last record.
CAMULATOR_TRANSIENT_END = datetime(2015, 1, 1)

# CESM2-LE (read from GDEX, not run - see cesm2leRunner) is offered for its
# historical runs only, 1850-2014, the years CAMulator's forcing follows.
CESM2LE_FIRST_START = datetime(1850, 1, 1)
CESM2LE_END = datetime(2015, 1, 1)

# DLESyM starts from ERA5, reaching back 48 hours, so from 3 January 1940.
# The Climate tab offers the 1st of each month through 2022.
DLESYM_FIRST_START = datetime(1940, 1, 3)
DLESYM_LAST_START = datetime(2022, 12, 1)


def climate_start_times(model):
    """The start times (00Z on the 1st of a month) the Climate tab offers
    for a model."""
    if model == 'CAMulator':
        return CAMULATOR_START_TIMES
    if model == 'CESM2-LE':
        return frozenset(datetime(y, m, 1)
                         for y in range(CESM2LE_FIRST_START.year, CESM2LE_END.year)
                         for m in range(1, 13))
    return frozenset(
        datetime(y, m, 1)
        for y in range(DLESYM_FIRST_START.year, DLESYM_LAST_START.year + 1)
        for m in range(1, 13)
        if DLESYM_FIRST_START <= datetime(y, m, 1) <= DLESYM_LAST_START)


def run_end_limit(model, forcing_mode=None):
    """The latest a run of the model may end, or None."""
    if model == 'CAMulator' and forcing_mode == 'transient':
        return CAMULATOR_TRANSIENT_END
    if model == 'CESM2-LE':
        return CESM2LE_END
    return None


# Models whose output is on a 0.25 degree grid interpolated from a native
# HEALPix grid of about 1 degree. They're saved every 4th point, on a
# 1 degree grid, which keeps all they resolve at 1/16th the size.
ONE_DEGREE_OUTPUT = {'DLESyMv0_ISCCP_ERA5'}

# Everything a climate model can write. A variable the Climate tab offers
# that its model doesn't have is left out, with a warning.
MODEL_OUTPUT_VARS = {
    'CAMulator': frozenset(
        [f'{v}{k}k' for v in ('u', 'v', 't', 'qtot') for k in range(32)]
        + ['sp', 't2m', 'tp06', 'skt', 'hcc', 'lcc', 'mcc', 'iews', 'inss',
           'ws10m', 'e06', 'msdwswrf', 'msdwlwrf', 'msuwswrf', 'msuwlwrf',
           'msshf', 'mslhf', 'mtuwswrf', 'mtuwlwrf']),
    'DLESyMv0_ISCCP_ERA5': frozenset(
        ['z500', 'z1000', 'z250', 't2m', 'tcwv', 't850', 'ws10m', 'rlut', 'sst']),
}

# ECMWF publishes IFS open data on three mirrors that throttle independently
# (AWS answered "503 Slow Down" until the fetch timed out, while Azure served
# the same files in seconds), so an IFS fetch tries each in turn. The ECMWF
# server only keeps the last 4 days; earth2studio rejects older times there
# up front, without a request.
#
# The mirrors are opened with cache=False. earth2studio's IFS cache names a
# file by time/variable/level only and downloads straight to that name, so
# a download cut off by a timeout stays behind as a truncated file that every
# later fetch - from any mirror, in any run - reads instead of refetching.
# Without the cache each mirror downloads to its own temporary directory,
# removed after each fetch. Only the initial state is fetched, so nothing is
# lost by not caching it.
IFS_MIRRORS = ['azure', 'aws', 'ecmwf']

MODEL_VAR_MAP = {
    'AIFS': {
        'U':    ['u50','u100','u150','u200','u250','u300','u400','u500','u600','u700','u850','u925','u1000'],
        'V':    ['v50','v100','v150','v200','v250','v300','v400','v500','v600','v700','v850','v925','v1000'],
        'T':    ['t50','t100','t150','t200','t250','t300','t400','t500','t600','t700','t850','t925','t1000'],
        'Q':    ['q50','q100','q150','q200','q250','q300','q400','q500','q600','q700','q850','q925','q1000'],
        'SP':   ['sp'],
        't2m':  ['t2m'],
        'U500': ['u500'],
        'V500': ['v500'],
        'T500': ['t500'],
        'Z500': ['z500'],
        'Q500': ['q500'],
    },
    'Aurora': {
        'U':    ['u50','u100','u150','u200','u250','u300','u400','u500','u600','u700','u850','u925','u1000'],
        'V':    ['v50','v100','v150','v200','v250','v300','v400','v500','v600','v700','v850','v925','v1000'],
        'T':    ['t50','t100','t150','t200','t250','t300','t400','t500','t600','t700','t850','t925','t1000'],
        'Q':    ['q50','q100','q150','q200','q250','q300','q400','q500','q600','q700','q850','q925','q1000'],
        'SP':   ['msl'],       # Aurora has msl, not sp
        't2m':  ['t2m'],
        'U500': ['u500'],
        'V500': ['v500'],
        'T500': ['t500'],
        'Z500': ['z500'],
        'Q500': ['q500'],
    },
    'Pangu': {
        'U':    ['u1000','u925','u850','u700','u600','u500','u400','u300','u250','u200','u150','u100','u50'],
        'V':    ['v1000','v925','v850','v700','v600','v500','v400','v300','v250','v200','v150','v100','v50'],
        'T':    ['t1000','t925','t850','t700','t600','t500','t400','t300','t250','t200','t150','t100','t50'],
        'Q':    ['q1000','q925','q850','q700','q600','q500','q400','q300','q250','q200','q150','q100','q50'],
        'SP':   ['msl'],       # Pangu has no surface pressure — uses mean sea level pressure, like Aurora
        't2m':  ['t2m'],
        'U500': ['u500'],
        'V500': ['v500'],
        'T500': ['t500'],
        'Z500': ['z500'],
        'Q500': ['q500'],
    },
    'FourCastNet3': {
        'U':    ['u50','u100','u150','u200','u250','u300','u400','u500','u600','u700','u850','u925','u1000'],
        'V':    ['v50','v100','v150','v200','v250','v300','v400','v500','v600','v700','v850','v925','v1000'],
        'T':    ['t50','t100','t150','t200','t250','t300','t400','t500','t600','t700','t850','t925','t1000'],
        'Q':    ['q50','q100','q150','q200','q250','q300','q400','q500','q600','q700','q850','q925','q1000'],
        'SP':   ['msl'],       # FCN3 has no surface pressure — uses mean sea level pressure, like Aurora
        't2m':  ['t2m'],
        'U500': ['u500'],
        'V500': ['v500'],
        'T500': ['t500'],
        'Z500': ['z500'],
        'Q500': ['q500'],
    },
    # CAMulator's output stays on CAM's 32 hybrid levels, u0k (model top)
    # to u31k (lowest). Q is its total water, vapour plus cloud
    # condensate. It has no geopotential.
    'CAMulator': {
        'U':    [f'u{k}k' for k in range(32)],
        'V':    [f'v{k}k' for k in range(32)],
        'T':    [f't{k}k' for k in range(32)],
        'Q':    [f'qtot{k}k' for k in range(32)],
        'SP':   ['sp'],
        't2m':  ['t2m'],
    },
    # DLESyM has no 3D fields or surface pressure; its OLR is rlut.
    'DLESyMv0_ISCCP_ERA5': {
        'U': [], 'V': [], 'T': [], 'Q': [], 'SP': [],
        'mtuwlwrf': ['rlut'],
    },
    # CESM2-LE is CAM6 on the same 32 levels and grid as CAMulator, so it
    # writes CAMulator's names. Its 6-hourly Q is vapour only, written as
    # qtot to compare with CAMulator's (cloud water is a tiny part of it).
    'CESM2-LE': {
        'U':    [f'u{k}k' for k in range(32)],
        'V':    [f'v{k}k' for k in range(32)],
        'T':    [f't{k}k' for k in range(32)],
        'Q':    [f'qtot{k}k' for k in range(32)],
        'SP':   ['sp'],
        't2m':  ['t2m'],
    },
    # GFS (fetched, not run - see forecastSourceRunner) has every field
    # the AI models write, so SP gives both sp and msl, to compare with
    # the models that have either.
    'GFS': {
        'U':    ['u50','u100','u150','u200','u250','u300','u400','u500','u600','u700','u850','u925','u1000'],
        'V':    ['v50','v100','v150','v200','v250','v300','v400','v500','v600','v700','v850','v925','v1000'],
        'T':    ['t50','t100','t150','t200','t250','t300','t400','t500','t600','t700','t850','t925','t1000'],
        'Q':    ['q50','q100','q150','q200','q250','q300','q400','q500','q600','q700','q850','q925','q1000'],
        'SP':   ['sp', 'msl'],
        't2m':  ['t2m'],
        'U500': ['u500'],
        'V500': ['v500'],
        'T500': ['t500'],
        'Z500': ['z500'],
        'Q500': ['q500'],
    },
    # GraphCast, SFNO — add their maps as you test them.
    # Fallback for unmapped models: pass vars through lowercased.
}

#CREDIT_TO_AIFS = {
#    'U':    ['u50','u100','u150','u200','u250','u300','u400','u500','u600','u700','u850','u925','u1000'],
#    'V':    ['v50','v100','v150','v200','v250','v300','v400','v500','v600','v700','v850','v925','v1000'],
#    'T':    ['t50','t100','t150','t200','t250','t300','t400','t500','t600','t700','t850','t925','t1000'],
#    'Q':    ['q50','q100','q150','q200','q250','q300','q400','q500','q600','q700','q850','q925','q1000'],
#    'SP':   ['sp'],
#    't2m':  ['t2m'],
#    'U500': ['u500'],
#    'V500': ['v500'],
#    'T500': ['t500'],
#    'Z500': ['z500'],
#    'Q500': ['q500'],
#}

class Earth2StudioRunner(ModelRunner):

    def validate(self, config) -> str | None:
        if not config["simulation_name"].strip():
            return "Error: Please enter a simulation name."
        selected = set(config["model"]) & set(MODEL_MAP.keys())
        if config["model"] not in MODEL_MAP:
            return f"Error: '{config['model']}' is not a recognized Earth2Studio model."
        if config["model"] == 'CAMulator':
            start = config["start_time"]
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            if start not in CAMULATOR_START_TIMES:
                return ("Error: CAMulator can only start at 00Z on 1 January or "
                        "1 July of 1980-2014 (1 January only in 1980), from "
                        f"the CAM states shipped with it; got {start:%Y-%m-%d %H}Z.")
            end = config["end_time"]
            if isinstance(end, str):
                end = datetime.fromisoformat(end)
            if config.get("forcing_mode") == 'transient' and end > CAMULATOR_TRANSIENT_END:
                return ("Error: CAMulator's transient forcing ends at "
                        f"{CAMULATOR_TRANSIENT_END:%Y-%m-%d %H}Z; this run ends "
                        f"{end:%Y-%m-%d %H}Z. Shorten it or use cyclic forcing.")
        elif config["model"] == 'DLESyMv0_ISCCP_ERA5':
            start = config["start_time"]
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            if start not in climate_start_times(config["model"]):
                return ("Error: DLESyM starts at 00Z on the 1st of a month, from "
                        f"{min(climate_start_times(config['model'])):%Y-%m} to "
                        f"{DLESYM_LAST_START:%Y-%m}; got {start:%Y-%m-%d %H}Z.")
        if config["model"] in MODEL_OUTPUT_VARS:
            model = config["model"]
            chosen = self._translateVars(config["ua_vars"] + config["surface_vars"], model)
            if not [v for v in chosen if v in MODEL_OUTPUT_VARS[model]]:
                return (f"Error: {model} writes none of the selected variables. "
                        "See its button's tooltip for what it has.")
#        if not selected:
#            return "Error: No recognized Earth2Studio model selected."
#        if len(selected) > 1:
#            return "Error: Please select only one Earth2Studio model at a time."
        return None

    def prepare(self, config) -> dict:
        return config

    def build_cmd(self, config) -> str:
        model_name = config["model"]

        python_bin = MODEL_ENV_MAP.get(model_name)
        if python_bin is None or not os.path.exists(python_bin):
            raise RuntimeError(
                f"No environment configured for model '{model_name}'. "
                f"Expected interpreter at: {python_bin}"
            )

        start      = config["start_time"]
        end        = config["end_time"]
        timestep   = config["timestep"]
        output_path = config["output_path"]
        file_stem  = config["file_stem"]
        ua_vars    = self._translateVars(config["ua_vars"], model_name)
        sfc_vars   = self._translateVars(config["surface_vars"], model_name)
        all_vars = list(dict.fromkeys(ua_vars + sfc_vars))
        # Selected variables the model doesn't write, reported in its log.
        missing = [label for label in config["ua_vars"] + config["surface_vars"]
                   if not self._translateVars([label], model_name)]
        if model_name in MODEL_OUTPUT_VARS:
            missing += [v for v in all_vars if v not in MODEL_OUTPUT_VARS[model_name]]
            all_vars = [v for v in all_vars if v in MODEL_OUTPUT_VARS[model_name]]
        missing_warning = (f"print('WARNING: {model_name} has no {', '.join(missing)}; "
                           "not saved.', flush=True)" if missing else "")

        # Compute number of steps from start/end/timestep
        hours = step_hours(timestep)
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        n_steps = int((end - start).total_seconds() / 3600 / hours)

        data_source = self._dataSource(model_name, start)
        if data_source == 'IFS':
            data_source_setup = (
                "data = MirroredSource('IFS', [(m, IFS(source=m, cache=False)) for m in "
                f"{IFS_MIRRORS!r}])")
        else:
            data_source_setup = f"data = {data_source}()"
        if data_source in LOCAL_DATA_SOURCES:
            data_source_import = (
                f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
                f"from {LOCAL_DATA_SOURCES[data_source]} import {data_source}")
        else:
            data_source_import = f"from earth2studio.data import {data_source}"

        # FCN3's DISCO convolutions overflow a 40 GB A100 (Derecho) unless
        # their weight contraction is chunked by latitude; 4 chunks matches
        # the unpatched output exactly. expandable_segments has to be set
        # before torch is imported.
        if model_name == 'FourCastNet3':
            alloc_env = 'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"'
            model_patch = (
                f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
                "import fcn3ChunkedDisco\n"
                "fcn3ChunkedDisco.apply(n_chunks=4)")
        # AIFS's graph-transformer message passing overflows a 40 GB A100
        # unless anemoi splits its edges into chunks, which it does at
        # inference when ANEMOI_INFERENCE_NUM_CHUNKS is set before anemoi is
        # imported. The chunks are summed, so the output is unchanged.
        elif model_name == 'AIFS':
            alloc_env = ('os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\n'
                         'os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = "16"')
            model_patch = ""
        else:
            alloc_env = model_patch = ""

        output_nc  = f"{output_path}/{file_stem}.nc"

        # CAMulator is driven by the forcing (cyclic or transient) chosen
        # on the Climate tab.
        if model_name == 'CAMulator':
            forcing_import = "from earth2studio.data import CAMulatorForcing"
            model_kwargs = (
                "forcing_data_source=CAMulatorForcing("
                f"mode={config.get('forcing_mode', 'cyclic')!r}, verbose=False)")
        else:
            forcing_import = model_kwargs = ""
        script_path = os.path.join(config["output_dir"], f"{file_stem}_run.py")

        output_coords = f'OrderedDict(variable=np.array({all_vars!r}))'
        if model_name in ONE_DEGREE_OUTPUT:
            output_coords = (f'OrderedDict(variable=np.array({all_vars!r}), '
                             'lat=np.linspace(90, -90, 181), lon=np.arange(360.0))')

        if model_name == 'DLESyMv0_ISCCP_ERA5':
            run_block = self._dlesym_run_block(start, n_steps, hours, output_coords)
        else:
            run_block = f"""deterministic(
    time=[datetime.fromisoformat("{start.isoformat()}")],
    nsteps={n_steps},
    prognostic=model,
    data=data,
    io=io,
    output_coords={output_coords},
)"""

        script = f"""
# Direct tqdm logging to stdout so messages get forwarded to the Output Log
import os
import sys
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cuda"
{alloc_env}

import logging
import torch
import importlib
import numpy as np
from collections import OrderedDict
from datetime import datetime
from earth2studio.io import NetCDF4Backend
from earth2studio.run import deterministic
{data_source_import}
{model_patch}
{forcing_import}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True
)

print("Loading model {model_name}...", flush=True)
model_module, model_class = "{MODEL_MAP[model_name]}".rsplit(".", 1)
mod = importlib.import_module(model_module)
ModelClass = getattr(mod, model_class)

print("Downloading/loading weights...", flush=True)
package = ModelClass.load_default_package()
model = ModelClass.load_model(package, {model_kwargs})

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {{device}}", flush=True)
model = model.to(device)

print("Setting up output backend...", flush=True)
io = NetCDF4Backend("{output_nc}", backend_kwargs={{'mode': 'w'}})

# Fetch from the first mirror that succeeds, starting each call with the
# one that last worked.
class MirroredSource:
    def __init__(self, name, mirrors):
        self.name = name
        self.mirrors = list(mirrors)    # [(label, source)]

    def __call__(self, time, variable):
        errors = []
        for i, (label, source) in enumerate(self.mirrors):
            try:
                da = source(time, variable)
            except Exception as e:
                print(f"WARNING: {{self.name}} fetch from {{label}} failed: {{e!r}}", flush=True)
                errors.append(f"{{label}}: {{e!r}}")
                continue
            if i:
                self.mirrors.insert(0, self.mirrors.pop(i))
            return da
        raise RuntimeError(f"Every {{self.name}} mirror failed - " + "; ".join(errors))

print("Setting up data source: {data_source}...", flush=True)
{data_source_setup}

{missing_warning}
print(f"Running inference: {n_steps} steps from {start.isoformat()}...", flush=True)
{run_block}
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

        #return f"python {script_path}"
        #return f"{EARTH2STUDIO_PYTHON} {script_path}"
        return f"{python_bin} {script_path}"

    def _dlesym_run_block(self, start, n_steps, hours, output_coords):
        """The rollout for DLESyM, which earth2studio's deterministic() can't
        run: each step is 4 days of 6-hourly atmosphere from 2 days of
        history, with the ocean's SST only valid every 48 hours. SST is
        held at its last valid value in between, so every saved lead time
        has all its variables."""
        return f"""from earth2studio.data.utils import fetch_data
from earth2studio.utils.coords import map_coords, split_coords

out_coords = {output_coords}
step = np.timedelta64({hours}, "h")
last_lead = {n_steps} * step
t0 = np.array([np.datetime64("{start.isoformat()}")])
ic = model.input_coords()
x, coords = fetch_data(source=data, time=t0, variable=ic["variable"],
                       lead_time=ic["lead_time"], device=device)
print("Fetched the initial state", flush=True)
io.add_array(OrderedDict(time=t0, lead_time=np.arange({n_steps} + 1) * step,
                         lat=out_coords["lat"], lon=out_coords["lon"]),
             out_coords["variable"])

ocean_step = model.ocean_output_times[0]
lead_dim = list(coords).index("lead_time")
sst = None

def save(x, coords):
    \"\"\"Write the lead times from 0 to the run's end, SST filled in.\"\"\"
    global sst
    leads = coords["lead_time"].astype("timedelta64[h]")
    s = list(coords["variable"]).index("sst")
    for j, lead in enumerate(leads):
        if lead % ocean_step == np.timedelta64(0, "h"):
            sst = x[:, j, s].clone()
        else:
            x[:, j, s] = sst
    keep = np.flatnonzero((leads >= np.timedelta64(0, "h")) & (leads <= last_lead))
    x = x.index_select(lead_dim, torch.as_tensor(keep, device=x.device))
    coords = coords.copy()
    coords["lead_time"] = coords["lead_time"][keep]
    x, coords = map_coords(x, coords, out_coords)
    io.write(*split_coords(x, coords))
    return leads[keep][-1]

iterator = model.create_iterator(x, coords)
save(*next(iterator))           # the initial state
n_chunks = -(-last_lead // model.atmos_output_times[-1])
for i in range(n_chunks):
    lead = save(*next(iterator))
    print(f"Step {{i + 1}}/{{n_chunks}}: saved through {{lead.astype(int)}} h", flush=True)"""

    def _dataSource(self, model_name, start):
        source = DEFAULT_DATA_SOURCE
        for begin, name in MODEL_DATA_SOURCE.get(model_name, []):
            if start >= begin:
                source = name
        return source

    def _translateVars(self, vars, model_name):
        var_map = MODEL_VAR_MAP.get(model_name, {})
        result = []
        for v in vars:
            if v in var_map:
                result.extend(var_map[v])
            else:
                result.append(v.lower())
        return list(dict.fromkeys(result))
