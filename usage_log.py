"""Usage logging: one JSON record per line, one file per session, in a
"drop box" directory that anyone can add files to but nobody can list or
tamper with others' files in.

Set the directory up once, as its owner:

    mkdir /glade/work/pearse/inferstudio_stats
    chmod 1733 /glade/work/pearse/inferstudio_stats

1733 lets other users create files there but not list the directory, and
the sticky bit stops them deleting or renaming anyone else's files. Each
session writes <session type>.<user>.<session job id>.jsonl (e.g.
dev.pearse.1234567.casper-pbs.jsonl), mode 644 so the owner can read it.
The session type leads the name, so dev and production files separate
with a glob. The developers' own sessions (OWN_DIR_USERS: dev, production
and local alike) go in a subdirectory named for the user instead, e.g.
pearse/dev.pearse.1234567.casper-pbs.jsonl, so testing doesn't count as
real use when metrics are gathered from the top level. Logging is off if
the directory doesn't exist, and with INFERSTUDIO_USAGE_LOG=0.

Every record has: ts (Unix seconds), user, session (the OOD session's PBS
job id, or "pid<n>" outside PBS), session_type (INFERSTUDIO_SESSION_TYPE,
set by bc_InferStudio: "dev" or "production"; "local" outside PBS), host,
event, and that event's own fields. Events:

    session_start    once per server: queue wait, requested resources
    page_open        each browser page load
    tab              main tab shown
    view             Visualization settings, once they've settled
    diff             a model difference selected: saved, computed or failed
    video_export     an export finished, was cancelled or failed
    stats_compute    a Compute Stats press on either Statistics sub-tab
    stats_region     credit-verify region picked
    inference_submit / inference_start / inference_end
                     a Run Inference press, its PBS job starting (with the
                     queue wait) and ending (with each model's outcome)

Nothing here ever raises: usage logging must never break the app.
"""

import getpass
import json
import os
import re
import socket
import subprocess
import threading
import time
from datetime import datetime

USAGE_DIR = os.environ.get("INFERSTUDIO_USAGE_DIR",
                           "/glade/work/pearse/inferstudio_stats")
ENABLED = os.environ.get("INFERSTUDIO_USAGE_LOG", "1") != "0"
# Users whose sessions are logged under USAGE_DIR/<user>/ rather than
# alongside everyone else's -- see the module docstring.
OWN_DIR_USERS = {"pearse"}

_lock = threading.Lock()
_session_logged = False


def _base():
    job = os.environ.get("PBS_JOBID")
    return {
        "user": getpass.getuser(),
        "session": job or f"pid{os.getpid()}",
        "session_type": os.environ.get(
            "INFERSTUDIO_SESSION_TYPE", "unknown" if job else "local"),
        "host": socket.gethostname(),
    }


def _path(base):
    name = f"{base['session_type']}.{base['user']}.{base['session']}.jsonl"
    if base["user"] in OWN_DIR_USERS:
        # Only that user writes here, so an ordinary 755 directory will do;
        # created on first use. The drop box itself must already exist --
        # its absence is what turns logging off.
        own = os.path.join(USAGE_DIR, base["user"])
        if os.path.isdir(USAGE_DIR):
            os.makedirs(own, mode=0o755, exist_ok=True)
        return os.path.join(own, name)
    return os.path.join(USAGE_DIR, name)


def log_event(event, **fields):
    """Append one record to this session's file."""
    if not ENABLED:
        return
    try:
        base = _base()
        record = {"ts": round(time.time(), 3), **base, "event": event, **fields}
        line = json.dumps(record, default=str) + "\n"
        with _lock:
            fd = os.open(_path(base), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.fchmod(fd, 0o644)  # whatever the user's umask, the owner can read it
                os.write(fd, line.encode())
            finally:
                os.close(fd)
    except Exception:
        pass


def short_error(exc, limit=300):
    """An exception or message, trimmed for a log record."""
    text = str(exc)
    return text if len(text) <= limit else text[:limit] + "..."


# ---------------------------------------------------------------------- #
#  The session itself                                                      #
# ---------------------------------------------------------------------- #

def _pbs_time(text):
    """Unix seconds from a qstat -f time such as 'Mon Oct 20 08:45:12 2026'."""
    try:
        return datetime.strptime(text.strip(), "%a %b %d %H:%M:%S %Y").timestamp()
    except ValueError:
        return None


def _session_job_info(job_id):
    """Queue wait and requested resources of the session's own PBS job."""
    from inference.pbsJob import _pbs
    out = subprocess.run([_pbs("qstat"), "-f", job_id], capture_output=True,
                         text=True, timeout=60).stdout
    attrs = dict(re.findall(r"^\s{4}(\S+) = (.*)$", out, re.M))
    info = {k: attrs.get(a) for k, a in (
        ("queue", "queue"),
        ("walltime", "Resource_List.walltime"),
        ("select", "Resource_List.select"),
        ("account", "Account_Name"),
    )}
    qtime = _pbs_time(attrs.get("qtime", ""))
    stime = _pbs_time(attrs.get("stime", ""))
    if qtime and stime:
        info["queue_wait_s"] = round(stime - qtime)
    return info


def log_session_start():
    """Once per server process, in the background (qstat can be slow)."""
    global _session_logged
    with _lock:
        if _session_logged or not ENABLED:
            return
        _session_logged = True

    def work():
        fields = {}
        job_id = os.environ.get("PBS_JOBID")
        if job_id:
            try:
                fields = _session_job_info(job_id)
            except Exception as e:
                fields = {"qstat_error": short_error(e)}
        log_event("session_start", **fields)

    threading.Thread(target=work, daemon=True, name="usage-session").start()


# ---------------------------------------------------------------------- #
#  Settled values                                                          #
# ---------------------------------------------------------------------- #

class Debounced:
    """Logs `event` once its fields have stopped changing for `delay`
    seconds, and only if they differ from what it logged last - so moving
    through several variables quickly logs the one the user stopped on."""

    def __init__(self, event, delay=3.0):
        self.event = event
        self.delay = delay
        self._timer = None
        self._last = None
        self._lock = threading.Lock()

    def __call__(self, **fields):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay, self._fire, kwargs=fields)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self, **fields):
        with self._lock:
            if fields == self._last:
                return
            self._last = fields
        log_event(self.event, **fields)
