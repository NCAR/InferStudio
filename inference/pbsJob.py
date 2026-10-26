"""Run inference in its own PBS batch job.

The InferStudio session itself only needs CPUs, so the models aren't run
in it. Run Inference writes a batch script that runs every selected model
and submits it with qsub, to Casper or Derecho, with GPUs the user picks on
the Inference tab. GLADE is mounted on both, so the job writes each
model's log and output where the app expects them, and the app follows
along by reading those files:

    <job dir>/job.sh            the batch script
    <job dir>/<model>.sh        the model's command
    <job dir>/started           when the job started (Unix seconds)
    <job dir>/<model>.start     when the model started
    <job dir>/<model>.exit      the model's exit status, once it finishes
    <job dir>/pbs.out           the job's own stdout/stderr

The job ends when its last model does. Its wall time is estimated from the
models and the forecast length (estimate_walltime), not asked for.
"""
import os
import re
import shlex
import subprocess
import sys

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Queue:
    queue: str              # queue@server, so it works from either system
    max_hours: int          # the longest wall time the queue allows
    whole_node: bool = False    # every job gets (and is charged for) a whole node


@dataclass(frozen=True)
class System:
    queues: dict            # label -> Queue; the first is the default
    gpu_types: dict         # label -> PBS gpu_type
    max_gpus: int           # GPUs per node
    ncpus_per_gpu: int
    mem_gb_per_gpu: int


# Casper's L40 nodes have 7 GPUs, its A100/H100 nodes 4 and its V100 nodes
# 4 or 8 (jobs are held to 4, like the rest); Derecho's GPU
# nodes have 4 A100 40GB. CPUs and memory are a per-GPU share of a node
# that leaves room for its other jobs. The first GPU type listed is the
# default; on Casper that's the A100 80GB, since AIFS runs out of memory
# on 40-45 GB cards.
#
# Derecho's main queue gives each job whole nodes, however few GPUs it
# asks for, and the job waits for a node with all 4 free. Its develop
# queue shares nodes, but allows only 6 hours (NCAR's queue docs).
SYSTEMS = {
    'Casper': System(
        queues={'casper': Queue('casper@casper-pbs', max_hours=24)},
        gpu_types={
            'NVIDIA A100 (80 GB)': 'a100_80gb',
            'NVIDIA H100 (80 GB)': 'h100_80gb',
            'NVIDIA L40 (45 GB)': 'l40',
            'NVIDIA V100 (32 GB)': 'v100',
        },
        max_gpus=4,
        ncpus_per_gpu=8,
        mem_gb_per_gpu=100,
    ),
    'Derecho': System(
        queues={
            'develop (shared)': Queue('develop@desched1', max_hours=6),
            'main (whole node)': Queue('main@desched1', max_hours=12, whole_node=True),
        },
        gpu_types={
            'NVIDIA A100 (40 GB)': 'a100',
        },
        max_gpus=4,
        ncpus_per_gpu=16,
        mem_gb_per_gpu=110,
    ),
}

# The same modules the OOD session loads (bc_InferStudio's script.sh.erb);
# both systems have them.
MODULES = "ncarenv/25.10 cuda/12.9.0 conda"

# Use PBS's own commands where they're installed: Casper's qstat on PATH is
# a caching wrapper that can be minutes behind.
PBS_BIN = '/opt/pbs/bin'

JOBS_DIR = Path.home() / '.inferstudio' / 'jobs'


def _pbs(cmd):
    path = os.path.join(PBS_BIN, cmd)
    return path if os.path.exists(path) else cmd


# ---------------------------------------------------------------------- #
#  Project account                                                         #
# ---------------------------------------------------------------------- #

def project_account():
    """The project the OOD session is charged to, or None.

    bc_InferStudio exports it as INFERSTUDIO_ACCOUNT. Failing that, it's
    read from this session's own PBS job."""
    account = os.environ.get('INFERSTUDIO_ACCOUNT', '').strip()
    if account:
        return account
    job_id = os.environ.get('PBS_JOBID')
    if not job_id:
        return None
    try:
        out = subprocess.run([_pbs('qstat'), '-f', job_id], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'^\s*Account_Name = (\S+)', out, re.M)
    return m.group(1) if m else None


def conda_env():
    """The conda environment the session runs in, which the job activates
    too: the MILES-CREDIT models run in it."""
    return (os.environ.get('INFERSTUDIO_CONDA_ENV')
            or os.environ.get('CONDA_PREFIX')
            or sys.prefix)


# ---------------------------------------------------------------------- #
#  Wall time                                                               #
# ---------------------------------------------------------------------- #

# Rough time for one model on one GPU: (fixed seconds, seconds per model
# step, hours per model step). The fixed part covers starting Python,
# loading the weights and fetching the initial state. Measured from earlier
# runs where there were any (FourCastNet3, CAMulator); the rest are
# guesses on the slow side.
RUN_TIME = {
    'AIFS':         (420, 20, 6),
    'Aurora':       (300, 8, 6),
    'Pangu':        (180, 3, 6),
    'GFS':          (60, 5, 6),     # a download, ~2 s per lead time measured
    'FourCastNet3': (180, 8, 6),
    'CAMulator':    (180, 1, 6),
    # A guess: about CAMulator's speed, but it fetches 2 days of ERA5 at
    # 0.25 degrees to start.
    'DLESyMv0_ISCCP_ERA5': (900, 1, 6),
    'WXFormer':     (600, 6, 1),
    'WXFormerGen2': (600, 6, 1),
}
DEFAULT_RUN_TIME = (600, 20, 6)

JOB_SETUP_SECONDS = 300     # loading modules, activating conda
SAFETY_FACTOR = 1.5         # first runs also download the weights
MIN_WALLTIME = 30 * 60
WALLTIME_STEP = 15 * 60     # round up to this


def model_seconds(model, start, end):
    """Estimated run time of one model, in seconds."""
    fixed, per_step, step_hours = RUN_TIME.get(model, DEFAULT_RUN_TIME)
    hours = max(0.0, (end - start).total_seconds() / 3600)
    return fixed + per_step * hours / step_hours


def assign_lanes(estimates, n_gpus):
    """Split the models between GPUs, longest first, each onto the GPU
    with the least work so far. Each GPU runs its models one after another.

    estimates: [(model, seconds)]. Returns a list of n_gpus model lists."""
    lanes = [[] for _ in range(n_gpus)]
    totals = [0.0] * n_gpus
    for model, seconds in sorted(estimates, key=lambda e: -e[1]):
        i = totals.index(min(totals))
        lanes[i].append(model)
        totals[i] += seconds
    return [lane for lane in lanes if lane]


def _longest_lane(estimates, n_gpus):
    seconds = dict(estimates)
    lanes = assign_lanes(estimates, max(1, n_gpus))
    return max((sum(seconds[m] for m in lane) for lane in lanes), default=0)


def estimate_walltime(estimates, n_gpus, queue, cpu_estimates=()):
    """Wall time to request, in seconds, and whether the queue's limit cut
    it short. cpu_estimates are runs that need no GPU; they share one lane
    running beside the GPU lanes."""
    longest = max(_longest_lane(estimates, n_gpus), sum(s for _, s in cpu_estimates))
    wall = (JOB_SETUP_SECONDS + longest) * SAFETY_FACTOR
    wall = max(MIN_WALLTIME, -(-wall // WALLTIME_STEP) * WALLTIME_STEP)
    limit = queue.max_hours * 3600
    return int(min(wall, limit)), wall > limit


# The default GPU count keeps each GPU busy at least this share of the
# time its longest lane runs: past that, another GPU is mostly paid for
# sitting idle.
MIN_GPU_BUSY = 0.75


def default_gpu_count(estimates, system, queue, cpu_estimates=()):
    """GPUs to ask for unless the user picks otherwise.

    On a whole-node queue the project pays for every GPU on the node, so
    use as many as the models can. Elsewhere, the most GPUs that stay busy
    (MIN_GPU_BUSY) - or, if that would hit the queue's wall-time limit,
    the fewest that don't."""
    most = max(1, min(SYSTEMS[system].max_gpus, len(estimates)))
    if queue.whole_node:
        return most
    work = sum(s for _, s in estimates)
    busy = [n for n in range(1, most + 1)
            if work >= MIN_GPU_BUSY * n * _longest_lane(estimates, n)]
    n = max(busy, default=1)
    while n < most and estimate_walltime(estimates, n, queue, cpu_estimates)[1]:
        n += 1
    return n


def format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------- #
#  The job                                                                 #
# ---------------------------------------------------------------------- #

@dataclass
class ModelRun:
    model: str
    cmd: str
    log_path: Path
    seconds: float
    gpu: bool = True        # False for a fetch (see forecastSourceRunner)


# A job with no GPU runs only downloads, so it goes to Casper whatever the
# System picker says: there it gets a few cores, where on Derecho it would
# be charged for a whole 128-core node.
CPU_ONLY_SYSTEM = 'Casper'
CPU_ONLY_QUEUE = next(iter(SYSTEMS[CPU_ONLY_SYSTEM].queues))
CPU_ONLY_SELECT = 'select=1:ncpus=2:mem=16GB'


@dataclass
class JobState:
    state: str              # PBS job_state (Q, H, R, E, F, ...), or '?'
    exec_host: str = ''
    exit_status: int = None
    comment: str = ''


@dataclass
class PBSJob:
    """One submitted inference job and the files it writes."""

    job_dir: Path
    system: str
    queue: str              # a key of SYSTEMS[system].queues
    job_id: str = ''
    models: list = field(default_factory=list)

    @classmethod
    def create(cls, name, system, queue):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', name)[:80]
        job_dir = JOBS_DIR / f"{safe}_{stamp}"
        job_dir.mkdir(parents=True, exist_ok=True)
        return cls(job_dir=job_dir, system=system, queue=queue)

    # --- files the job writes --- #

    def started_at(self):
        return self._read_number(self.job_dir / 'started')

    def model_started_at(self, model):
        return self._read_number(self.job_dir / f'{model}.start')

    def model_exit(self, model):
        code = self._read_number(self.job_dir / f'{model}.exit')
        return None if code is None else int(code)

    def model_ended_at(self, model):
        """When the model finished: the .exit file is written the moment it
        does, so its modification time is the end time."""
        try:
            return (self.job_dir / f'{model}.exit').stat().st_mtime
        except OSError:
            return None

    def pbs_output(self):
        try:
            return (self.job_dir / 'pbs.out').read_text(errors='replace')
        except OSError:
            return ''

    @staticmethod
    def _read_number(path):
        try:
            text = path.read_text().strip()
        except OSError:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    # --- submitting --- #

    def write_script(self, runs, account, gpu_type, n_gpus, walltime, workdir):
        """Write the batch script that runs `runs` ([ModelRun]) on n_gpus
        GPUs, and return its path. Runs that need no GPU share one more
        lane, with no GPU; with only those, the job asks for no GPU."""
        system = SYSTEMS[self.system]
        self.models = [r.model for r in runs]
        gpu_runs = [r for r in runs if r.gpu]
        cpu_lane = [r.model for r in runs if not r.gpu]
        lanes = assign_lanes([(r.model, r.seconds) for r in gpu_runs], n_gpus)
        n_gpus = len(lanes)
        by_model = {r.model: r for r in runs}

        for r in runs:
            (self.job_dir / f'{r.model}.sh').write_text(r.cmd + '\n')

        d = shlex.quote(str(self.job_dir))
        h, rem = divmod(int(walltime), 3600)
        if n_gpus:
            select = (f"select=1:ncpus={system.ncpus_per_gpu * n_gpus}"
                      f":mem={system.mem_gb_per_gpu * n_gpus}GB"
                      f":ngpus={n_gpus}:gpu_type={gpu_type}")
        else:
            select = CPU_ONLY_SELECT
        job_name = 'InferStudio-' + '-'.join(self.models)
        job_name = re.sub(r'[^A-Za-z0-9_-]', '_', job_name)[:60]

        lines = [
            '#!/bin/bash -l',
            f'#PBS -N {job_name}',
            f'#PBS -A {account}',
            f'#PBS -q {system.queues[self.queue].queue}',
            f'#PBS -l {select}',
            f'#PBS -l walltime={h:02d}:{rem // 60:02d}:00',
            '#PBS -j oe',
            f'#PBS -o {self.job_dir / "pbs.out"}',
            '',
            f'JOB_DIR={d}',
            'date +%s > "$JOB_DIR/started"',
            'echo "InferStudio inference job $PBS_JOBID on $(hostname)"',
            '',
            f'module load {MODULES}',
            'eval "$(conda shell.bash hook)"',
            f'conda activate {shlex.quote(conda_env())}',
            f'cd {shlex.quote(str(workdir))}',
            'export PYTHONUNBUFFERED=1',
            '',
            '# PBS lists the GPUs it gave the job in CUDA_VISIBLE_DEVICES;',
            '# each lane below gets one of them to itself.',
            f'IFS=, read -ra GPUS <<< "${{CUDA_VISIBLE_DEVICES:-{",".join(map(str, range(n_gpus)))}}}"',
            '',
            '# run_model <model> <gpu> <log>: writes when the model started and,',
            '# once it ends, its exit status.',
            'run_model() {',
            '    date +%s > "$JOB_DIR/$1.start"',
            '    CUDA_VISIBLE_DEVICES="$2" bash "$JOB_DIR/$1.sh" > "$3" 2>&1',
            '    echo $? > "$JOB_DIR/$1.exit"',
            '}',
            '',
        ]
        for i, lane in enumerate(lanes):
            calls = '; '.join(
                f'run_model {m} "${{GPUS[{i}]}}" {shlex.quote(str(by_model[m].log_path))}'
                for m in lane)
            lines.append(f'( {calls} ) &')
        if cpu_lane:
            calls = '; '.join(
                f'run_model {m} "" {shlex.quote(str(by_model[m].log_path))}'
                for m in cpu_lane)
            lines.append(f'( {calls} ) &')
        lines += [
            'wait',
            'echo "All models finished."',
            '',
        ]
        script = self.job_dir / 'job.sh'
        script.write_text('\n'.join(lines))
        return script

    def submit(self, script):
        """qsub the script. Raises RuntimeError with qsub's message on
        failure."""
        try:
            result = subprocess.run([_pbs('qsub'), str(script)], capture_output=True,
                                    text=True, timeout=120, cwd=self.job_dir)
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"qsub failed: {e}") from e
        if result.returncode != 0:
            raise RuntimeError("qsub failed: " + (result.stderr or result.stdout).strip())
        self.job_id = result.stdout.strip()
        return self.job_id

    # --- following it --- #

    def status(self):
        """The job's state from qstat, or None if PBS doesn't know it
        (anymore)."""
        try:
            result = subprocess.run([_pbs('qstat'), '-x', '-f', self.job_id],
                                    capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return JobState('?')
        if result.returncode != 0:
            if 'Unknown Job Id' in result.stderr:
                return None
            return JobState('?')
        fields = {}
        for line in result.stdout.splitlines():
            m = re.match(r'\s+(\w+) = (.*)', line)
            if m:
                fields[m.group(1)] = m.group(2).strip()
        exit_status = fields.get('Exit_status')
        return JobState(
            state=fields.get('job_state', '?'),
            exec_host=fields.get('exec_host', '').split('/')[0],
            exit_status=int(exit_status) if exit_status not in (None, '') else None,
            comment=fields.get('comment', ''),
        )

    def delete(self):
        """qdel the job. Returns qdel's error message, or None."""
        if not self.job_id:
            return None
        try:
            result = subprocess.run([_pbs('qdel'), self.job_id], capture_output=True,
                                    text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            return str(e)
        if result.returncode != 0:
            return (result.stderr or result.stdout).strip()
        return None


class LogTail:
    """Reads what has been added to a file since the last read."""

    def __init__(self, path):
        self.path = Path(path)
        self._pos = 0

    def read(self):
        try:
            with open(self.path, 'r', errors='replace') as f:
                f.seek(self._pos)
                text = f.read()
                self._pos = f.tell()
            return text
        except OSError:
            return ''

