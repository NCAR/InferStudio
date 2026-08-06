# Per-model environments

This page is for maintainers and for users diagnosing a run that fails on import.
Ordinary use of InferStudio does not require touching any of it.

## Why one environment per model

The models cannot share a Python environment. The conflicts are not stylistic:

- Pangu needs a GPU ONNX Runtime build; the PyTorch models do not use ONNX at
  all.
- SFNO and FourCastNet 3 need `torch-harmonics` compiled against a specific
  PyTorch ABI. Two models needing different PyTorch versions therefore need
  different builds of the same extension.
- AIFS needs `flash_attn`, which itself compiles against a specific PyTorch and
  CUDA architecture.
- Earth2Studio's own dependency resolver will happily upgrade PyTorch past the
  CUDA version supported by the host driver, producing an environment that
  installs cleanly and fails at the first kernel launch.

The resolution is isolation: one [uv](https://docs.astral.sh/uv/) virtual
environment per model, each pinned independently.

## Layout

```text
/glade/work/$USER/E2S/envs/
├── pangu/        bin/python  → Python 3.12
├── aifs/         bin/python
├── aurora/       bin/python
├── sfno/         bin/python
└── fcn3/         bin/python
```

The application's `MODEL_ENV_MAP` maps each model name to the corresponding
`bin/python`. Nothing else in the app needs to know these environments exist.

## Site constraints

Two constraints drive most of the pins:

**Driver CUDA ceiling.** Casper's GPU driver supports up to CUDA 12.7. Any
PyTorch wheel built against a newer CUDA toolkit will import and then fail at
runtime. Pin PyTorch explicitly; do not let a transitive resolver choose it.

**GPU architecture.** The L40 nodes are compute capability 8.9. Compiled
extensions must be built with `TORCH_CUDA_ARCH_LIST="8.9"` or you will get
`no kernel image is available for execution on the device`.

## Known-good build notes

:::{admonition} These are snapshots, not a supported contract
:class: warning

Upstream moves. The pins below were valid at the time of writing and are recorded
because reconstructing them is expensive, not because they are guaranteed to work
against current upstream `main`.
:::

### Pangu

```bash
uv pip install onnxruntime-gpu==1.21.0
```

The CPU `onnxruntime` package will be picked up in preference to the GPU one if
both are present. Install only the GPU package.

### SFNO

Requires `makani` from GitHub plus a source build of `torch-harmonics`:

```bash
uv pip install "makani @ git+https://github.com/NVIDIA/modulus-makani"
uv pip install numpy==2.4.6 warp-lang==1.12.1 nvidia-physicsnemo==1.0.1

export FORCE_CUDA_EXTENSION=1
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_HOME=$NCAR_ROOT_CUDA        # after: module load cuda/12.9.0
uv pip install torch-harmonics==0.9.1 --no-build-isolation --no-deps
```

`--no-build-isolation` is required so the build sees the already-installed
PyTorch; `--no-deps` prevents the build from dragging in a newer PyTorch.

### FourCastNet 3

FCN3 needs a **matched pair** of commits — `makani` at `29f5bc4` with
`torch-harmonics` at `a4ac667`. Mixing versions produces import errors that look
like ABI problems but are API drift.

Additionally: a git-installed `torch-harmonics` reports its version as `0.0.0`,
which causes `makani`'s `_cuda_extension_available` check to conclude the CUDA
extension is absent and fall back to a slow path. The workaround is a monkeypatch
forcing that check to return `True`.

```bash
uv pip install triton==3.2.0 numpy==2.4.6 nvidia-physicsnemo==1.0.1
```

### AIFS

```bash
uv pip install flash_attn --no-build-isolation
```

Expect a long compile. Build it on a compute node, not a login node.

## Validating an environment

Before wiring a new environment into `MODEL_ENV_MAP`, check three things in
order. Each failure mode is distinct and there is no point testing the next until
the previous passes.

1. **Import** — does the model class import at all?

   ```bash
   /glade/work/$USER/E2S/envs/<model>/bin/python -c \
     "from earth2studio.models.px import <Model>; print('import ok')"
   ```

2. **Load** — does the checkpoint load and move to the GPU?

   ```bash
   /glade/work/$USER/E2S/envs/<model>/bin/python -c "
   from earth2studio.models.px import <Model>
   m = <Model>.load_model(<Model>.load_default_package()).to('cuda')
   print('load ok')"
   ```

3. **Inference** — one step from a real initial condition. This needs an
   exclusive GPU; a shared one will produce out-of-memory errors that mask real
   problems.

## Diagnosing a failure from the log

| Symptom | Likely cause |
| --- | --- |
| `undefined symbol: _ZN...` on import | Extension compiled against a different PyTorch than the one installed |
| `no kernel image is available` | Extension built without `TORCH_CUDA_ARCH_LIST="8.9"` |
| `CUDA driver version is insufficient` | PyTorch built against a CUDA newer than the driver's 12.7 ceiling |
| Silently slow inference | `makani` fell back to a non-CUDA path — check the `_cuda_extension_available` issue above |
| `ModuleNotFoundError` for a model dependency | `--no-deps` build skipped something genuinely needed; install it explicitly |
