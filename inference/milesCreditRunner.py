import importlib.util
import os
import shlex

from inference.modelRunner import ModelRunner
from pathlib import Path

class MilesCreditRunner(ModelRunner):

    def validate(self, config) -> str | None:
        if not config["simulation_name"].strip():
            return "Error: Please enter a simulation name."
        return None

    def prepare(self, config) -> dict:
        with open('model_predict_casper.yml', 'r') as f:
            # The GFS initial conditions go under the user's own scratch
            # ($USER in the config).
            content = os.path.expandvars(f.read())

        content = content.replace(
            'forecast_start_time: "2025-07-02 00:00:00"',
            f'forecast_start_time: "{config["start_time"]}"'
        )
        content = content.replace(
            'forecast_end_time: "2025-07-02 02:00:00"',
            f'forecast_end_time: "{config["end_time"]}"'
        )
        # The model always steps forward its own 1 hour, whatever the output
        # frequency: rollout_realtime makes one model step per forecast_timestep,
        # so a 6h timestep would label 1-hour steps as 6 hours apart. The hours
        # between outputs are removed after the run instead (see build_cmd).

        content = content.replace(
            "save_vars: []",
            f"save_vars: {config['surface_vars'] + config['ua_vars']}"
        )
        #content = content.replace(
        #    "variables: ['U','V','T','Q']",
        #    f"variables: {config['ua_vars']}"
        #)
        #content = content.replace(
        #    "surface_variables: ['SP','t2m','V500','U500','T500','Z500','Q500']",
        #    f"surface_variables: {config['surface_vars']}"
        #)

        content = content.replace(
            "save_forecast: '/glade/derecho/scratch/pearse/CREDIT/RAW_OUTPUT/wxformer_1h_gfs_demo/'",
            f"save_forecast: '{config['output_path']}'"
        )

        config_file = config["output_dir"] + '/' + config["file_stem"] + '.yml'
        with open(config_file, 'w') as f:
            f.write(content)

        return {"config_file": config_file}

    def build_cmd(self, config) -> str:
        cfg = config["config_file"]
        return (
            f"python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/gfs_init.py -c {cfg} &&"
            f"python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/rollout_realtime.py -c {cfg}"
            + _thin_cmd(config)
        )


# The WXFormer config for MILES-CREDIT's gen2 pipeline. Static: the init
# time, step count and output directory are passed on the command line.
GEN2_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "model_predict_gen2.yml")
WXFORMER_STEP_HOURS = 1     # WXFormer's own time step
# Runs rollout_realtime_gen2, reading GFS from Google Cloud (see the module).
GEN2_ROLLOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "creditGen2Rollout.py")

# The checkout this runs from, whose cf_convert gathers gen2's output into
# one file.
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Both WXFormer rollouts (gen1's rollout_realtime, stepping 1 hour at a time,
# and rollout_realtime_gen2) save every step, as
# <save_dir>/<init>Z/pred_<init>Z_<FHR:03d>.nc. This removes the hours that
# aren't a multiple of the chosen output frequency afterwards.
_THIN = """
import glob, os, re, sys
save_dir, every = sys.argv[1], int(sys.argv[2])
for f in glob.glob(os.path.join(save_dir, '*Z', 'pred_*Z_*.nc')):
    fhr = int(re.search(r'_(\\d+)\\.nc$', f).group(1))
    if fhr % every:
        os.remove(f)
"""


def _thin_cmd(config) -> str:
    """The command (with a leading " && ") that removes the hours between
    outputs, or "" if every hour is output."""
    every = int(config["timestep"].rstrip("h"))
    if every <= WXFORMER_STEP_HOURS:
        return ""
    return f" && python -c {shlex.quote(_THIN)} {shlex.quote(config['output_path'])} {every}"


def _cf_cmd(config) -> str:
    """The command (with a leading " && ") that gathers the saved hours
    into <file_stem>_cf.nc, the one file per model the Visualization tab
    reads (see cf_convert.wxformer_to_cf)."""
    save_dir = config["output_path"]
    out = os.path.join(save_dir, f"{config['file_stem']}_cf.nc")
    return (f" && python {shlex.quote(os.path.join(APP_DIR, 'cf_convert.py'))} --wxformer "
            f"{shlex.quote(save_dir)} {shlex.quote(out)}")


class MilesCreditGen2Runner(ModelRunner):
    """WXFormer through MILES-CREDIT's gen2 realtime rollout
    (credit.applications.rollout_realtime_gen2, via creditGen2Rollout), which
    starts from a GFS analysis it fetches itself."""

    def validate(self, config) -> str | None:
        if not config["simulation_name"].strip():
            return "Error: Please enter a simulation name."
        try:
            has_gen2 = importlib.util.find_spec("credit.applications.rollout_realtime_gen2") is not None
        except ModuleNotFoundError:     # no miles-credit at all
            has_gen2 = False
        if not has_gen2:
            return ("Error: WXFormer (gen 2) needs miles-credit 2026.2 or later, which "
                    "this conda environment doesn't have. Launch InferStudio with the "
                    "InferStudio environment.")
        if self._steps(config) < 1:
            return "Error: The forecast must be at least one hour long."
        return None

    def prepare(self, config) -> dict:
        return {}

    def build_cmd(self, config) -> str:
        save_dir = config["output_path"]
        return (
            f"python {shlex.quote(GEN2_ROLLOUT)} -c {shlex.quote(GEN2_CONFIG)}"
            f" --init-time {config['start_time']:%Y-%m-%dT%H}"
            f" --steps {self._steps(config)}"
            f" --save-dir {shlex.quote(save_dir)}"
            + _thin_cmd(config)
            + _cf_cmd(config)
        )

    @staticmethod
    def _steps(config) -> int:
        hours = (config["end_time"] - config["start_time"]).total_seconds() / 3600
        return int(hours // WXFORMER_STEP_HOURS)
