"""
Run MILES-CREDIT's gen2 realtime rollout (rollout_realtime_gen2), working
around bugs in it as of miles-credit 2026.2.0 (also on upstream main as of
2026-10-15):

1. For an init time less than 10 days old it reads the GDAS analysis from
   NOMADS over HTTP, but credit.nwp opens every file with the GCS-only
   storage option token="anon", which the HTTP filesystem passes on to
   aiohttp and fails on ("unexpected keyword argument 'token'"). Recent
   analyses are also in the gs://global-forecast-system bucket, which gen1's
   gfs_init.py always reads from, so this does the same.
2. It hands the preblocks a batch keyed source-first ({"era5": {"input": ...}})
   and reads their result's "input", but the preblocks take a batch keyed
   data-type-first ({"input": {"ERA5": ...}}) and return the concatenated
   tensor as "x" ("No 'input' tensors found in batch.").
3. After each step it feeds the prediction back in physical units, into an
   input the model expects normalized.
4. It applies the tracer fixer's thresholds (e.g. Q >= 1e-8 kg/kg) to the
   normalized prediction, where they'd clip humidity at roughly its mean.
   They're converted to normalized units here.

Takes rollout_realtime_gen2's own arguments:

    python creditGen2Rollout.py -c model_predict_gen2.yml \
        --init-time 2026-10-14T00 --steps 6 --save-dir /path/to/output
"""

from credit.applications import rollout_realtime_gen2 as rr

GCS_GFS = "gs://global-forecast-system/"

_build_GFS_init = rr.build_GFS_init
_apply_preblocks = rr.apply_preblocks
_update_x = rr.update_x
_inject_tracer_inds = rr._inject_tracer_inds

# The output normalization (mean, std), shaped (1, C_out, 1, 1, 1), set once
# the tracer fixer is configured, which happens before the rollout starts.
_denorm = {}


def _build_GFS_init_from_gcs(*args, **kwargs):
    kwargs["gdas_base_path"] = GCS_GFS
    print(f"Reading the GFS analysis from {GCS_GFS}", flush=True)
    return _build_GFS_init(*args, **kwargs)


def _sample_to_batch(sample):
    # Variable keys are "<source>/<field type>/<dim>/<name>". The metadata is
    # left out: the preblocks don't need it, and its datetimes are plain ints
    # here rather than the tensors ConcatToTensor expects.
    batch = {"input": {}}
    for key, tensor in sample["input"].items():
        source = key.split("/")[0]
        batch["input"].setdefault(source, {})[key] = tensor.unsqueeze(0)
    return batch


def _apply_preblocks_as_input(*args, **kwargs):
    out = _apply_preblocks(*args, **kwargs)
    return {**out, "input": out["x"]}


def _update_x_normalized(x_prev, x_dynfrc, y_pred, groups):
    mean, std = (t.to(y_pred.device, y_pred.dtype) for t in _denorm["stats"])
    return _update_x(x_prev, x_dynfrc, (y_pred - mean) / std, groups)


def _inject_tracer_inds_normalized(conf):
    _inject_tracer_inds(conf)
    mean, std = rr._build_output_denorm(conf, "cpu")
    _denorm["stats"] = (mean, std)
    tracer_conf = conf.get("model", {}).get("post_conf", {}).get("tracer_fixer", {})
    if tracer_conf.get("activate", False):
        mean, std = mean.flatten(), std.flatten()
        tracer_conf["tracer_thres"] = [
            float((float(thres) - mean[i]) / std[i])
            for i, thres in zip(tracer_conf["tracer_inds"], tracer_conf["tracer_thres"])
        ]


rr.build_GFS_init = _build_GFS_init_from_gcs
rr._sample_to_batch = _sample_to_batch
rr.apply_preblocks = _apply_preblocks_as_input
rr.update_x = _update_x_normalized
rr._inject_tracer_inds = _inject_tracer_inds_normalized

if __name__ == "__main__":
    rr.main()
