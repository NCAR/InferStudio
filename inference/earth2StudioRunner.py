from inference.modelRunner import ModelRunner
from datetime import datetime, timedelta
import os

#EARTH2STUDIO_PYTHON = f"/glade/work/{os.environ['USER']}/conda-envs/earth2studio/bin/python"
#EARTH2STUDIO_PYTHON = f"/glade/work/pearse/conda-envs/earth2studio/bin/python"
MODEL_ENV_MAP = {
    'AIFS':         f"/glade/work/pearse/E2S/envs/aifs/bin/python",
    'Aurora':       f"/glade/work/pearse/E2S/envs/aurora/bin/python",
    'FourCastNet3': f"/glade/work/pearse/E2S/envs/fcn3/bin/python",
    'GraphCast':    f"/glade/work/pearse/E2S/envs/graphcast/bin/python",
    'Pangu':        f"/glade/work/pearse/E2S/envs/pangu/bin/python",
    'SFNO':         f"/glade/work/pearse/E2S/envs/sfno/bin/python",
}

MODEL_MAP = {
    'AIFS':         'earth2studio.models.px.AIFS',
    'Aurora':       'earth2studio.models.px.Aurora',
    'FourCastNet3': 'earth2studio.models.px.FCN3',      # was FourCastNetv2Small — wrong model entirely
    'GraphCast':    'earth2studio.models.px.GraphCastSmall',
    'Pangu':        'earth2studio.models.px.Pangu6',
    'SFNO':         'earth2studio.models.px.SFNO',
}

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
    # FourCastNet3, GraphCast, Pangu, SFNO — add their maps as you test them.
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
        sim_name   = config["simulation_name"]
        ua_vars    = self._translateVars(config["ua_vars"], model_name)
        sfc_vars   = self._translateVars(config["surface_vars"], model_name)
        all_vars = list(dict.fromkeys(ua_vars + sfc_vars))

        # Compute number of steps from start/end/timestep
        hours = int(timestep.replace("h", ""))
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        n_steps = int((end - start).total_seconds() / 3600 / hours)

        output_nc  = f"{output_path}/{sim_name}.nc"
        #script_path = os.path.join(config["output_dir"], f"{sim_name}_run.py")
        script_path = os.path.join(config["output_dir"], f"{sim_name}_{model_name}_run.py")


        script = f"""
# Direct tqdm logging to stdout so messages get forwarded to the Output Log
import os
import sys
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cuda"

import logging
import torch
import importlib
import numpy as np
from datetime import datetime
from earth2studio.io import NetCDF4Backend
from earth2studio.run import deterministic
from earth2studio.data import GFS

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
model = ModelClass.load_model(package)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {{device}}", flush=True)
model = model.to(device)

print("Setting up output backend...", flush=True)
io = NetCDF4Backend("{output_nc}", backend_kwargs={{'mode': 'w'}})

print("Setting up data source...", flush=True)
data = GFS()

print(f"Running inference: {n_steps} steps from {start.isoformat()}...", flush=True)
deterministic(
    time=[datetime.fromisoformat("{start.isoformat()}")],
    nsteps={n_steps},
    prognostic=model,
    data=data,
    io=io,
    output_coords={{
        "variable": np.array({all_vars}),
    }},
)
print("Output written to {output_nc}", flush=True)

print("Converting to CF-compliant format...", flush=True)
try:
    sys.path.insert(0, "/glade/work/pearse/InferStudio")
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

    def _translateVars(self, vars, model_name):
        var_map = MODEL_VAR_MAP.get(model_name, {})
        result = []
        for v in vars:
            if v in var_map:
                result.extend(var_map[v])
            else:
                result.append(v.lower())
        return list(dict.fromkeys(result))
