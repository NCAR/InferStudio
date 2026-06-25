from modelRunner import ModelRunner
from pathlib import Path

class MilesCreditRunner(ModelRunner):

    def validate(self, config) -> str | None:
        if not config["simulation_name"].strip():
            return "Error: Please enter a simulation name."
        return None

    def prepare(self, config) -> dict:
        with open('model_predict_casper.yml', 'r') as f:
            content = f.read()

        content = content.replace(
            'forecast_start_time: "2025-07-02 00:00:00"',
            f'forecast_start_time: "{config["start_time"]}"'
        )
        content = content.replace(
            'forecast_end_time: "2025-07-02 02:00:00"',
            f'forecast_end_time: "{config["end_time"]}"'
        )
        content = content.replace(
            'forecast_timestep: "1h"',
            f'forecast_timestep: "{config["timestep"]}"'
        )

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

        config_file = config["output_dir"] + '/' + config["simulation_name"] + '.yml'
        with open(config_file, 'w') as f:
            f.write(content)

        return {"config_file": config_file}

    def build_cmd(self, config) -> str:
        cfg = config["config_file"]
        return (
            f"python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/gfs_init.py -c {cfg} &&"
            f"python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/rollout_realtime.py -c {cfg}"
        )
