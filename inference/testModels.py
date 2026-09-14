#!/usr/bin/env python3
"""
testModels.py — exercise Earth2Studio models one at a time, outside InferStudio.

The parent process runs under any stdlib Python. For each model it re-executes
THIS file with that model's venv interpreter plus --child, so a broken env only
takes down its own subprocess.

Stages are cumulative:
    import  -> import the model class
    package -> load_default_package() (may download weights)
    load    -> load_model(package)
    device  -> model.to(cuda)
    coords  -> dump input/output coords, emit a MODEL_VAR_MAP suggestion
    infer   -> short deterministic() run against GFS into a temp NetCDF

Usage:
    python testModels.py --list
    python testModels.py                             # all models, through coords
    python testModels.py -m Pangu SFNO -v
    python testModels.py -m FourCastNet3 -s infer --steps 1
    python testModels.py -m GraphCast -s load -v
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timedelta, timezone

MODEL_ENV_MAP = {
    'AIFS':         "/glade/work/pearse/E2S/envs/aifs/bin/python",
    'Aurora':       "/glade/work/pearse/E2S/envs/aurora/bin/python",
    'FourCastNet3': "/glade/work/pearse/E2S/envs/fcn3/bin/python",
    'GraphCast':    "/glade/work/pearse/E2S/envs/graphcast/bin/python",
    'Pangu':        "/glade/work/pearse/E2S/envs/pangu/bin/python",
    'SFNO':         "/glade/work/pearse/E2S/envs/sfno/bin/python",
}

MODEL_MAP = {
    'AIFS':         'earth2studio.models.px.AIFS',
    'Aurora':       'earth2studio.models.px.Aurora',
    'FourCastNet3': 'earth2studio.models.px.FCN3',
    'GraphCast':    'earth2studio.models.px.GraphCastSmall',
    'Pangu':        'earth2studio.models.px.Pangu6',
    'SFNO':         'earth2studio.models.px.SFNO',
}

STAGES = ["import", "package", "load", "device", "coords", "infer"]
RESULT_PREFIX = "@@RESULT@@ "
CHILD_ENV = {
    "TQDM_DISABLE": "1",
    "PYTHONUNBUFFERED": "1",
    "JAX_PLATFORMS": "cuda",
}
PROBE_VARS = ["t2m", "z500", "t500", "u500", "v500"]


def main(argv=None):
    args = _parse_args(argv)
    if args.child:
        return run_child(args)
    if args.list:
        return list_envs()
    return run_parent(args)


def list_envs():
    print(f"{'MODEL':<14} {'CLASS':<40} INTERPRETER")
    for name in sorted(MODEL_MAP):
        python_bin = MODEL_ENV_MAP.get(name, "")
        mark = "ok " if python_bin and os.path.exists(python_bin) else "MISSING"
        print(f"{name:<14} {MODEL_MAP[name]:<40} [{mark}] {python_bin or '(unconfigured)'}")
    return 0


def run_parent(args):
    models = args.models or sorted(MODEL_MAP)
    os.makedirs(args.outdir, exist_ok=True)
    reports = {}

    for name in models:
        print(f"\n{'=' * 72}\n{name} -> stage '{args.stage}'\n{'=' * 72}", flush=True)
        report = _run_one(name, args)
        reports[name] = report

        report_path = os.path.join(args.outdir, f"{name}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        for stage in STAGES:
            entry = report["stages"].get(stage)
            if entry is None:
                continue
            status = "PASS" if entry["ok"] else "FAIL"
            print(f"  {stage:<8} {status}  ({entry['seconds']:.1f}s)", flush=True)
            if not entry["ok"]:
                print(_indent(entry.get("error", "")), flush=True)

        if report.get("output_variables"):
            print(f"\n  {len(report['output_variables'])} output variables; "
                  f"suggested MODEL_VAR_MAP entry:", flush=True)
            print(_format_var_map(name, suggest_var_map(report["output_variables"])), flush=True)

        if report.get("grid"):
            print(f"  grid: {report['grid']}", flush=True)
        print(f"  full report: {report_path}", flush=True)

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    failed = 0
    for name, report in reports.items():
        reached = [s for s in STAGES if report["stages"].get(s, {}).get("ok")]
        broke = next((s for s in STAGES
                      if s in report["stages"] and not report["stages"][s]["ok"]), None)
        if broke or not reached:
            failed += 1
            detail = f"failed at '{broke}'" if broke else (report.get("error") or "no stages ran")
            print(f"  {name:<14} FAIL  {detail}")
        else:
            print(f"  {name:<14} PASS  through '{reached[-1]}'")
    return 1 if failed else 0


def suggest_var_map(variables):
    """Derive a canonical CREDIT -> model variable mapping from a variable list."""
    present = list(variables)
    lookup = set(present)
    level_pattern = re.compile(r"^([uvtqz])(\d+)$")
    by_prefix = {"u": [], "v": [], "t": [], "q": [], "z": []}

    for var in present:
        match = level_pattern.match(var)
        if match:
            by_prefix[match.group(1)].append(var)

    suggestion = {}
    for canonical, prefix in (("U", "u"), ("V", "v"), ("T", "t"), ("Q", "q")):
        if by_prefix[prefix]:
            suggestion[canonical] = by_prefix[prefix]

    if "sp" in lookup:
        suggestion["SP"] = ["sp"]
    elif "msl" in lookup:
        suggestion["SP"] = ["msl"]

    for canonical, candidate in (("t2m", "t2m"), ("U500", "u500"), ("V500", "v500"),
                                 ("T500", "t500"), ("Z500", "z500"), ("Q500", "q500")):
        if candidate in lookup:
            suggestion[canonical] = [candidate]
    return suggestion


def run_child(args):
    report = {"model": args.model, "stages": {}, "error": None}
    target = STAGES.index(args.stage)
    state = {}

    for stage in STAGES[:target + 1]:
        handler = _CHILD_STAGES[stage]
        start = datetime.now()
        try:
            handler(args, state, report)
            ok, error = True, None
        except Exception:
            ok, error = False, traceback.format_exc()
        report["stages"][stage] = {
            "ok": ok,
            "seconds": (datetime.now() - start).total_seconds(),
            "error": error,
        }
        if not ok:
            break

    print(RESULT_PREFIX + json.dumps(report), flush=True)
    return 0 if all(s["ok"] for s in report["stages"].values()) else 1


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--models", nargs="+", choices=sorted(MODEL_MAP),
                        help="models to test (default: all)")
    parser.add_argument("-s", "--stage", choices=STAGES, default="coords",
                        help="last stage to attempt (default: coords)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream child stdout/stderr as it arrives")
    parser.add_argument("--steps", type=int, default=1,
                        help="nsteps for the infer stage (default: 1)")
    parser.add_argument("--start", default=None,
                        help="ISO start time for infer (default: 00Z two days ago)")
    parser.add_argument("--vars", nargs="+", default=None,
                        help="explicit model-native variables for the infer stage")
    parser.add_argument("--outdir", default="./modelTestReports",
                        help="where per-model JSON reports are written")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="seconds before a child is killed (default: 3600)")
    parser.add_argument("--list", action="store_true",
                        help="show configured interpreters and exit")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.start is None:
        stamp = datetime.now(timezone.utc) - timedelta(days=2)
        args.start = stamp.replace(hour=0, minute=0, second=0,
                                   microsecond=0, tzinfo=None).isoformat()
    return args


def _run_one(name, args):
    report = {"model": name, "stages": {}, "error": None}
    python_bin = MODEL_ENV_MAP.get(name)
    if not python_bin or not os.path.exists(python_bin):
        report["error"] = f"no interpreter at {python_bin!r}"
        print(f"  {report['error']}", flush=True)
        return report

    cmd = [
        python_bin, os.path.abspath(__file__), "--child",
        "--model", name,
        "--stage", args.stage,
        "--steps", str(args.steps),
        "--start", args.start,
        "--outdir", os.path.abspath(args.outdir),
    ]
    if args.vars:
        cmd += ["--vars"] + args.vars

    env = dict(os.environ)
    env.update(CHILD_ENV)
    env.pop("PYTHONPATH", None)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    killer = threading.Timer(args.timeout, proc.kill)
    killer.start()
    payload = None
    try:
        for line in proc.stdout:
            if line.startswith(RESULT_PREFIX):
                payload = line[len(RESULT_PREFIX):].strip()
            elif args.verbose:
                print(f"  | {line.rstrip()}", flush=True)
        proc.wait()
    finally:
        killer.cancel()

    if payload:
        report.update(json.loads(payload))
    elif proc.returncode == -9:
        report["error"] = f"killed after {args.timeout}s"
    else:
        report["error"] = f"child exited {proc.returncode} without a result payload"
    return report


def _indent(text, prefix="      "):
    return "\n".join(prefix + line for line in text.rstrip().splitlines())


def _format_var_map(name, suggestion):
    lines = [f"    '{name}': {{"]
    for key, values in suggestion.items():
        rendered = ",".join(f"'{v}'" for v in values)
        lines.append(f"        {key + ':':<7} [{rendered}],")
    lines.append("    },")
    return "\n".join(lines)


def _stage_import(args, state, report):
    import importlib
    module_path, class_name = MODEL_MAP[args.model].rsplit(".", 1)
    print(f"importing {module_path}.{class_name}", flush=True)
    state["ModelClass"] = getattr(importlib.import_module(module_path), class_name)
    report["earth2studio_version"] = getattr(
        __import__("earth2studio"), "__version__", "unknown")


def _stage_package(args, state, report):
    print("resolving default package (may download weights)", flush=True)
    state["package"] = state["ModelClass"].load_default_package()


def _stage_load(args, state, report):
    print("loading model", flush=True)
    state["model"] = state["ModelClass"].load_model(state["package"])


def _stage_device(args, state, report):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["device"] = device
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    print(f"moving model to {device}", flush=True)
    state["model"] = state["model"].to(device)


def _stage_coords(args, state, report):
    model = state["model"]
    input_coords = model.input_coords()
    report["input_variables"] = [str(v) for v in input_coords["variable"]]
    report["grid"] = {
        key: int(len(value))
        for key, value in input_coords.items()
        if key in ("lat", "lon", "time", "lead_time")
    }
    report["lon_range"] = [float(input_coords["lon"].min()),
                           float(input_coords["lon"].max())]
    try:
        output_coords = model.output_coords(input_coords)
        report["output_variables"] = [str(v) for v in output_coords["variable"]]
        report["lead_time"] = [str(v) for v in output_coords["lead_time"]]
    except Exception as exc:
        report["output_variables"] = report["input_variables"]
        report["output_coords_error"] = repr(exc)
        print(f"output_coords() failed, falling back to input list: {exc}", flush=True)
    print(f"{len(report['output_variables'])} output variables", flush=True)


def _stage_infer(args, state, report):
    import numpy as np
    from earth2studio.data import GFS
    from earth2studio.io import NetCDF4Backend
    from earth2studio.run import deterministic

    available = report.get("output_variables") or report.get("input_variables") or []
    if args.vars:
        requested = list(args.vars)
    else:
        requested = [v for v in PROBE_VARS if v in available] or available[:3]
    missing = [v for v in requested if available and v not in available]
    report["infer_requested"] = requested
    report["infer_missing"] = missing
    if missing:
        raise RuntimeError(f"requested variables not offered by model: {missing}")

    out_nc = os.path.join(args.outdir, f"{args.model}_probe.nc")
    io = NetCDF4Backend(out_nc, backend_kwargs={"mode": "w"})
    print(f"running {args.steps} step(s) from {args.start} for {requested}", flush=True)
    try:
        deterministic(
            time=[datetime.fromisoformat(args.start)],
            nsteps=args.steps,
            prognostic=state["model"],
            data=GFS(),
            io=io,
            output_coords={"variable": np.array(requested)},
        )
    finally:
        if hasattr(io, "close"):
            io.close()

    report["infer_output"] = out_nc
    report["infer_bytes"] = os.path.getsize(out_nc)
    try:
        from netCDF4 import Dataset
        with Dataset(out_nc) as ds:
            report["infer_written_vars"] = sorted(ds.variables)
            report["infer_dims"] = {k: len(v) for k, v in ds.dimensions.items()}
    except Exception as exc:
        report["infer_inspect_error"] = repr(exc)
    print(f"wrote {out_nc} ({report['infer_bytes']} bytes)", flush=True)


_CHILD_STAGES = {
    "import": _stage_import,
    "package": _stage_package,
    "load": _stage_load,
    "device": _stage_device,
    "coords": _stage_coords,
    "infer": _stage_infer,
}


if __name__ == "__main__":
    sys.exit(main())
