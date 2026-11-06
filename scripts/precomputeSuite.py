#!/usr/bin/env python
"""Compute ahead of time what the Visualization and Statistics tabs
otherwise compute when asked, for one simulation suite:

- every difference between two plottable models, both ways round (A minus
  B and B minus A), as the Visualization tab's Compute Difference writes
  them (modelDiff.compute_model_difference);
- both kinds of verification statistics against GFS analysis (the basic
  RMSE/CRPS and the WeatherBench2 scores) for every variable and level of
  every model the Statistics tab can verify, saved where it looks for them
  (forecastStatsPanel.save_stats).

Each is saved where the tab looks for it, so opening the suite shows it
straight away. Anything already saved and current is left as it is, so
this can be rerun. One that fails is reported and skipped; the tab can
still try it again.

Usage (from the app directory, in the InferStudio conda environment):

    python scripts/precomputeSuite.py <suite_dir>
    python scripts/precomputeSuite.py <suite_dir> --no-stats
"""

import argparse
import os
import sys
import time
from datetime import datetime
from itertools import permutations
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
# datasetPlot reads static/styles.css relative to the app directory.
os.chdir(APP_DIR)

import xarray as xr

from dimensions import PRES_NAME, diff_file_path, resolve_nc_glob, vertical_dim
from visualization.datasetPlot import EARTH2STUDIO_FORMAT_MODELS as PLOTTABLE_MODELS
from visualization.earth2StudioVars import parse_variable_groups
from visualization.forecastStatsPanel import (
    EARTH2STUDIO_FORMAT_MODELS as VERIFIABLE_MODELS, STATS_DIRNAME, STATS_VERSION,
    basic_scores, compute_model_stats, load_saved_stats, save_stats)
from visualization.modelDiff import compute_model_difference
from visualization import wb2StatsPanel

# The Statistics tab's two methods: (name, saved under, version, score, columns).
STATS_METHODS = [
    ('basic', STATS_DIRNAME, STATS_VERSION, basic_scores, ('rmse', 'crps')),
    ('credit-verify', wb2StatsPanel.WB2StatsPanel.STATS_SUBDIR,
     wb2StatsPanel.WB2StatsPanel.STATS_VERSION, wb2StatsPanel.wb2_scores,
     wb2StatsPanel.COLUMNS),
]


def suite_models(suite_dir, wanted):
    return sorted(p.name for p in suite_dir.iterdir() if p.is_dir() and p.name in wanted)


def compute_diffs(suite_dir):
    """Every A minus B. Returns the number that failed."""
    failed = 0
    for a, b in permutations(suite_models(suite_dir, PLOTTABLE_MODELS), 2):
        if diff_file_path(suite_dir / a, a, b).exists():
            continue
        started = time.time()
        try:
            compute_model_difference(suite_dir / a, suite_dir / b, a, b)
        except Exception as e:
            log(f"{a} minus {b}: failed: {e}")
            failed += 1
            continue
        log(f"{a} minus {b}: {time.time() - started:.0f} s")
    return failed


def stats_fields(model_dir):
    """The (variable, level) pairs the Statistics tab can pick for a model:
    each pressure level of a leveled variable, and level 0 (as the tab's
    level_value has it) for a surface one."""
    with xr.open_mfdataset(resolve_nc_glob(model_dir), engine='netcdf4',
                           data_vars='all') as ds:
        level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))
        fields = [(base, level) for base, levels in level_vars.items() for level in levels]
        for name in surface_vars:
            dim = vertical_dim(ds[name])
            if dim == PRES_NAME:
                fields += [(name, int(level)) for level in ds[name][PRES_NAME].values]
            elif dim is None:
                fields.append((name, 0))
    return sorted(fields)


def compute_stats(suite_dir):
    """Both methods' stats for every field of every verifiable model.
    Returns the number of (model, field, method)s that failed."""
    failed = 0
    for model in suite_models(suite_dir, VERIFIABLE_MODELS):
        model_dir = suite_dir / model
        for var, level in stats_fields(model_dir):
            label = f"{model} {var}" + (f" {level} hPa" if level else "")
            for method, subdir, version, score, columns in STATS_METHODS:
                saved = load_saved_stats(model_dir, var, level, subdir, version)
                reuse = saved[0] if saved else None
                if reuse is not None and reuse[columns[0]].notna().all():
                    continue
                started = time.time()
                try:
                    df = compute_model_stats(model_dir, var, level, reuse=reuse,
                                             score=score, columns=columns)
                    save_stats(model_dir, var, level, df, subdir, version)
                except Exception as e:
                    log(f"{label} ({method}): failed: {e}")
                    failed += 1
                    continue
                bad = df[df[columns[0]].isna()]
                if len(bad):
                    log(f"{label} ({method}): {len(bad)} of {len(df)} lead times "
                        f"failed, e.g. {bad['error'].iloc[0]}")
                    failed += 1
                else:
                    log(f"{label} ({method}): {time.time() - started:.0f} s")
    return failed


def log(text):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('suite_dir', type=Path)
    parser.add_argument('--no-diffs', action='store_true')
    parser.add_argument('--no-stats', action='store_true')
    args = parser.parse_args(argv)

    suite_dir = args.suite_dir.resolve()
    failed = 0
    if not args.no_diffs:
        failed += compute_diffs(suite_dir)
    if not args.no_stats:
        failed += compute_stats(suite_dir)
    log(f"Done; {failed} failed." if failed else "Done.")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
