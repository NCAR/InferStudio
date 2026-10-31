#!/usr/bin/env python
"""Run the example suites: every weather model in one suite, every climate
model in another, each saving every variable the Inference tab offers.

Run nightly from cron (see nightlyExamples.sh). Each suite is submitted as
one PBS job, built the same way the Inference tab's Run Inference builds it
(the same runners, GPU count and wall time), into a staging directory. If
every model succeeds, a second, CPU-only job (post.sh in the same job
directory) computes every model difference and all the verification
statistics (scripts/precomputeSuite.py), then makes the staged suite the
published one. If any model fails, PBS drops that second job: the published
suite is left as it was and the staged one stays behind to look at:

    <EXAMPLES_DIR>/WeatherExample               the published suites
    <EXAMPLES_DIR>/ClimateExample
    <EXAMPLES_DIR>/.staging/<suite>/<suite>     tonight's, while it runs
    <EXAMPLES_DIR>/.staging/<suite>/jobid       its PBS job

A suite whose last jobs are still queued or running isn't submitted again.

Usage (from the app directory, in the InferStudio conda environment, with
INFERSTUDIO_ACCOUNT set):

    python scripts/nightlyExamples.py                  # both suites
    python scripts/nightlyExamples.py -s Weather
    python scripts/nightlyExamples.py --dry-run        # write the job scripts, don't qsub
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from dimensions import model_file_stem
from inference.cesm2leRunner import CESM2LE_MEMBERS, CESM2LERunner
from inference.earth2StudioRunner import Earth2StudioRunner
from inference.forecastSourceRunner import ForecastSourceRunner
from inference.milesCreditRunner import MilesCreditGen2Runner
from inference.pbsJob import (CPU_ONLY_QUEUE, CPU_ONLY_SYSTEM, MODULES, SYSTEMS,
                              ModelRun, PBSJob, _pbs, conda_env, default_gpu_count,
                              estimate_walltime, model_seconds, project_account)

EXAMPLES_DIR = Path('/glade/campaign/cisl/vast/pearse/InferStudioExamples')

# The nightly jobs' directories (one per suite per night, with each
# model's job script, PBS output and post.out), beside the cron log
# nightlyExamples.sh appends to. Those of earlier nights are removed after
# KEEP_JOB_DIRS.
NIGHTLY_DIR = Path('/glade/campaign/cisl/vast/pearse/inferStudio/logs/nightly')
KEEP_JOB_DIRS = timedelta(days=14)

# Where the GPU models run: the Inference tab's defaults.
SYSTEM = 'Derecho'
QUEUE = next(iter(SYSTEMS[SYSTEM].queues))
GPU_TYPE = next(iter(SYSTEMS[SYSTEM].gpu_types.values()))

# The job that computes differences and statistics after the models: on
# Derecho's develop queue too, since PBS only chains jobs on the same
# server, but with no GPU, as a shared node's share.
POST_QUEUE = SYSTEMS[SYSTEM].queues[QUEUE].queue
POST_SELECT = 'select=1:ncpus=8:mem=64GB'
# Measured: 25 minutes for the weather suite (15 of them statistics, mostly
# fetching GFS), 4 for the climate one.
POST_WALLTIME = '01:30:00'

# Every variable each tab's buttons offer (OutputParams' defaults on the
# Weather tab, inferenceTab.CLIMATE_SURFACE_VARS on the Climate tab; copied,
# since importing those brings in Panel). The tab only offers a variable
# every selected model writes; here each model gets all of them, and its
# runner leaves out the ones it doesn't write, with a warning in its log.
WEATHER_UA_VARS = ['U', 'V', 'T', 'Q']
WEATHER_SURFACE_VARS = ['SP', 't2m']
CLIMATE_SURFACE_VARS = ['SP', 't2m', 'skt', 'ws10m', 'mtuwlwrf', 'hcc', 'mcc', 'lcc',
                        'z500', 't850', 'tcwv', 'sst']


def weather_suite():
    # The Weather tab's default forecast: 3 steps of 6 hours, from the
    # cycle 24 hours ago.
    now = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
    start = now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0)
    return dict(
        name='WeatherExample',
        runners=[('WXFormerGen2', MilesCreditGen2Runner()),
                 ('AIFS', Earth2StudioRunner()),
                 ('Aurora', Earth2StudioRunner()),
                 ('Pangu', Earth2StudioRunner()),
                 ('FourCastNet3', Earth2StudioRunner()),
                 ('GFS', ForecastSourceRunner())],
        config=dict(start_time=start, end_time=start + timedelta(hours=18), timestep='6h',
                    ua_vars=WEATHER_UA_VARS, surface_vars=WEATHER_SURFACE_VARS),
    )


def climate_suite():
    # The Climate tab's defaults: a month from January 2000, cyclic forcing.
    # DLESyM is left out for now.
    return dict(
        name='ClimateExample',
        runners=[('CAMulator', Earth2StudioRunner()),
                 ('CESM2-LE', CESM2LERunner())],
        config=dict(start_time=datetime(2000, 1, 1), end_time=datetime(2000, 2, 1),
                    timestep='6h', ua_vars=WEATHER_UA_VARS,
                    surface_vars=CLIMATE_SURFACE_VARS,
                    forcing_mode='cyclic', member=CESM2LE_MEMBERS[0]),
    )


SUITES = {'Weather': weather_suite, 'Climate': climate_suite}


def submit_suite(suite, account, dry_run=False):
    """Stage and submit one suite. Returns True if it was submitted (or,
    with dry_run, written)."""
    name = suite['name']
    stage = EXAMPLES_DIR / '.staging' / name
    if still_running(stage / 'jobid'):
        log(f"{name}: last night's job {(stage / 'jobid').read_text().strip()} "
            "is still in PBS; not submitting another.")
        return False
    shutil.rmtree(stage, ignore_errors=True)
    sim_dir = stage / name
    sim_dir.mkdir(parents=True)

    runs, failed = [], []
    for model, runner in suite['runners']:
        model_dir = sim_dir / model
        model_dir.mkdir()
        config = dict(suite['config'], model=model, simulation_name=name,
                      file_stem=model_file_stem(name, model),
                      # The run's scripts and caches go beside the suite,
                      # not in it: every directory in it is read as a model.
                      output_dir=str(stage), output_path=str(model_dir))
        error = runner.validate(config)
        try:
            if error:
                raise RuntimeError(error)
            config.update(runner.prepare(config))
            cmd = runner.build_cmd(config)
        except Exception as e:
            log(f"{name}: {model} not run: {e}")
            failed.append(model)
            continue
        gpu = not isinstance(runner, (ForecastSourceRunner, CESM2LERunner))
        runs.append(ModelRun(model, cmd, model_dir / f'{model}_run.log',
                             model_seconds(model, config['start_time'], config['end_time']),
                             gpu=gpu))
    if not runs:
        log(f"{name}: no model could be set up; not submitted.")
        return False

    gpu_estimates = [(r.model, r.seconds) for r in runs if r.gpu]
    cpu_estimates = [(r.model, r.seconds) for r in runs if not r.gpu]
    system, queue = (SYSTEM, QUEUE) if gpu_estimates else (CPU_ONLY_SYSTEM, CPU_ONLY_QUEUE)
    queue_spec = SYSTEMS[system].queues[queue]
    n_gpus = (default_gpu_count(gpu_estimates, system, queue_spec, cpu_estimates)
              if gpu_estimates else 0)
    walltime, capped = estimate_walltime(gpu_estimates, n_gpus, queue_spec, cpu_estimates)
    if capped:
        log(f"{name}: needs more than {system} {queue}'s wall-time limit; it may be cut short.")

    job = PBSJob.create(name, system, queue, jobs_dir=NIGHTLY_DIR)
    script = job.write_script(runs, account, GPU_TYPE, n_gpus, walltime, APP_DIR)
    # Any model failing to set up already means an incomplete suite.
    with open(script, 'a') as f:
        f.write(check_lines(name, [r.model for r in runs], ok=not failed))
    post = write_post_script(job, name, account)
    log(f"{name}: {', '.join(r.model for r in runs)} on {system} {queue}, "
        f"{n_gpus} GPU(s), {walltime // 60} min; job files in {job.job_dir}")
    if dry_run:
        log(f"{name}: dry run, not submitted: {script}, then {post}")
        return True
    try:
        job_id = job.submit(script)
    except RuntimeError as e:
        log(f"{name}: {e}")
        return False
    (stage / 'jobid').write_text(job_id + '\n')
    log(f"{name}: submitted {job_id}")
    try:
        post_id = qsub_after(post, job_id, job.job_dir)
    except RuntimeError as e:
        log(f"{name}: {e}; the suite won't be published.")
        return False
    # The last job of the two: still in PBS while either is.
    (stage / 'jobid').write_text(post_id + '\n')
    log(f"{name}: submitted {post_id} to follow it")
    return True


def check_lines(name, models, ok):
    """Lines for the end of the inference job script: fail the job unless
    every model succeeded, so the job that publishes the suite doesn't
    run."""
    stage = shlex.quote(str(EXAMPLES_DIR / '.staging' / name / name))
    checks = ' && '.join(f'[ "$(cat "$JOB_DIR/{m}.exit" 2>/dev/null)" = 0 ]' for m in models)
    return f"""
if ! {{ {'true' if ok else 'false'} && {checks}; }}; then
    echo "Not every model succeeded; {name} left as it was. This run: {stage}"
    exit 1
fi
"""


def write_post_script(job, name, account):
    """The job that follows the models: precompute the staged suite's
    differences and statistics, then make it the published one, readable
    by everyone. It's published even if some of those failed; the tabs
    compute anything missing when asked."""
    stage = shlex.quote(str(EXAMPLES_DIR / '.staging' / name / name))
    published = shlex.quote(str(EXAMPLES_DIR / name))
    old = shlex.quote(str(EXAMPLES_DIR / f'.{name}.old'))
    script = job.job_dir / 'post.sh'
    script.write_text(f"""#!/bin/bash -l
#PBS -N InferStudio-{name}-post
#PBS -A {account}
#PBS -q {POST_QUEUE}
#PBS -l {POST_SELECT}
#PBS -l walltime={POST_WALLTIME}
#PBS -j oe
#PBS -o {job.job_dir / 'post.out'}

echo "InferStudio {name} differences and statistics, job $PBS_JOBID on $(hostname)"
module load {MODULES}
eval "$(conda shell.bash hook)"
conda activate {shlex.quote(conda_env())}
cd {shlex.quote(str(APP_DIR))}
export PYTHONUNBUFFERED=1
python scripts/precomputeSuite.py {stage} || echo "Some weren't computed; publishing anyway."

chmod -R a+rX {stage}
rm -rf {old}
if [ -e {published} ]; then mv {published} {old}; fi
mv {stage} {published} && rm -rf {old}
echo "Published {name} to {published}."
""")
    return script


def qsub_after(script, job_id, cwd):
    """qsub the script to run once job_id has ended successfully; PBS
    deletes it if job_id fails. Returns its job id."""
    try:
        result = subprocess.run([_pbs('qsub'), '-W', f'depend=afterok:{job_id}', str(script)],
                                capture_output=True, text=True, timeout=120, cwd=cwd)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"qsub failed: {e}") from e
    if result.returncode != 0:
        raise RuntimeError("qsub failed: " + (result.stderr or result.stdout).strip())
    return result.stdout.strip()


def still_running(jobid_file):
    try:
        job_id = jobid_file.read_text().strip()
    except OSError:
        return False
    state = PBSJob(job_dir=jobid_file.parent, system='', queue='', job_id=job_id).status()
    # None: PBS no longer knows it. '?': qstat failed, so assume it's
    # still there rather than clobber its staging directory.
    return state is not None and state.state not in ('F', 'X')


def prune_job_dirs():
    cutoff = time.time() - KEEP_JOB_DIRS.total_seconds()
    for path in NIGHTLY_DIR.glob('*Example_*'):
        if path.name.split('_')[0] in ('WeatherExample', 'ClimateExample'):
            try:
                if path.stat().st_mtime < cutoff:
                    shutil.rmtree(path)
            except OSError:
                pass


def log(text):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--suites', nargs='+', choices=list(SUITES), default=list(SUITES))
    parser.add_argument('--dry-run', action='store_true',
                        help="write the job scripts but don't submit them")
    args = parser.parse_args(argv)

    account = project_account()
    if not account:
        log("No project account: set INFERSTUDIO_ACCOUNT.")
        return 1
    os.chdir(APP_DIR)
    os.umask(0o022)
    prune_job_dirs()
    ok = [submit_suite(SUITES[s](), account, args.dry_run) for s in args.suites]
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
