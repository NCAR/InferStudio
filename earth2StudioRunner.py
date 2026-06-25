from modelRunner import ModelRunner
from datetime import datetime, timedelta
import os

MODEL_MAP = {
    'AIFS':         'earth2studio.models.px.AIFS',
    'Aurora':       'earth2studio.models.px.Aurora',
    'FourCastNet3': 'earth2studio.models.px.FourCastNetv2Small',
    'GraphCast':    'earth2studio.models.px.GraphCastSmall',
    'Pangu':        'earth2studio.models.px.Pangu6',
    'SFNO':         'earth2studio.models.px.SFNO',
}

CREDIT_TO_AIFS = {
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
}

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
        try:
            import earth2studio  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "earth2studio is not installed in this environment. "
                "Run: pip install earth2studio"
            )

        model_name = config["model"]
        #model_name = (set(config["model"]) & set(MODEL_MAP.keys())).pop()
        start      = config["start_time"]
        end        = config["end_time"]
        timestep   = config["timestep"]
        output_path = config["output_path"]
        sim_name   = config["simulation_name"]
        ua_vars    = self._translateVars(config["ua_vars"])
        sfc_vars   = self._translateVars(config["surface_vars"])
        all_vars = list(dict.fromkeys(ua_vars + sfc_vars))

        # Compute number of steps from start/end/timestep
        hours = int(timestep.replace("h", ""))
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        n_steps = int((end - start).total_seconds() / 3600 / hours)

        output_nc  = f"{output_path}/{sim_name}.nc"
        script_path = os.path.join(config["output_dir"], f"{sim_name}_run.py")

        script = f"""
# Direct tqdm logging to stdout so messages get forwarded to the Output Log
import os
import sys
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

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
"""

        with open(script_path, 'w') as f:
            f.write(script)

        return f"python {script_path}"

    def _translateVars(self, vars):
        result = []
        for v in vars:
            if v in CREDIT_TO_AIFS:
                result.extend(CREDIT_TO_AIFS[v])
            else:
                result.append(v.lower())  # fallback: just lowercase it
        return list(dict.fromkeys(result))  # deduplicate while preserving order
    
