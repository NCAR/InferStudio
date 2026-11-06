#!/bin/bash -l
# Submit the nightly example suites (see nightlyExamples.py). Meant for
# cron on cron.hpc.ucar.edu, whose environment is nearly empty, so
# everything the driver needs is set here. Crontab entry:
#
#   0 0 * * * /glade/work/pearse/ProductionInferStudio/scripts/nightlyExamples.sh >> /glade/campaign/cisl/vast/pearse/inferStudio/logs/nightly/nightlyExamples.log 2>&1

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

# The project the jobs are charged to, and the environment they (and the
# driver) run in.
export INFERSTUDIO_ACCOUNT=NVST0001
export INFERSTUDIO_CONDA_ENV=/glade/work/pearse/conda-envs/InferStudio

echo "=== $(date) on $(hostname): $APP_DIR"
cd "$APP_DIR" || exit 1
exec "$INFERSTUDIO_CONDA_ENV/bin/python" scripts/nightlyExamples.py "$@"
