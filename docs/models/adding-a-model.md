# Adding a model

Adding a model that Earth2Studio already supports is a four-step job. Adding a
model from a new framework means writing a runner.

## Adding an Earth2Studio model

### 1. Build the environment

Create an isolated uv venv and pin it against the site constraints described in
{doc}`environments`:

```bash
cd /glade/work/$USER/E2S/envs
uv venv --python 3.12 <model>
source <model>/bin/activate
uv pip install earth2studio
# plus whatever the model needs — see the model's upstream docs
```

Validate import, load, and one inference step before going further. A model that
is registered in the app but whose environment is broken produces a confusing
failure for every user who selects it.

### 2. Register the interpreter

Add an entry to `MODEL_ENV_MAP` in the Earth2Studio runner module:

```python
MODEL_ENV_MAP = {
    ...
    "<model>": "/glade/work/pearse/E2S/envs/<model>/bin/python",
}
```

### 3. Confirm the generated script is correct

The runner builds the inference script as a string and passes it to the child
interpreter. Two things to check for a new model:

- The correct `earth2studio.models.px` class is imported.
- `earth2studio.run.deterministic()` is given an **explicit data source** — it
  does not default to one. `GFS()` is the usual choice:

  ```python
  from earth2studio.data import GFS
  from earth2studio.io import NetCDF4Backend

  run.deterministic([time], nsteps, model, GFS(), io)
  ```

:::{warning}
Because the script is generated inside an f-string, edits to it must go *inside*
the f-string, and any literal braces in the generated code must be doubled. A
change made just outside the f-string will appear correct in the source and have
no effect on the child process. Verify with `grep -n` on the file after editing.
:::

### 4. Confirm output detection

Run the model and load its output in the dataset browser. Confirm that the
convention is detected as Earth2Studio (a `lead_time` dimension is present) and
that the variable list shows flattened names. If detection is wrong, the plotting
controls will be built for the wrong convention — see
{doc}`../reference/output-formats`.

## Adding a new model family

If the model is not driven by Earth2Studio or CREDIT, subclass the abstract
runner base:

```python
class MyFrameworkRunner(ModelRunner):
    """Dispatch inference for <framework> models."""

    def build_command(self, config) -> list[str]:
        """Return argv for the child process."""

    def parse_status(self, line: str):
        """Map a line of child output to a UI status update."""

    def output_path(self, config) -> Path:
        """Where the run will write its results."""
```

Requirements the base class expects you to honour:

- Launch with `start_new_session=True` so cancellation can signal the process
  group.
- Stream stdout and stderr line-by-line rather than buffering to completion — the
  log tab is the user's only view into a long run.
- **Never mutate a Panel or Bokeh widget from the reader thread.** Route updates
  through the document's next-tick callback. Direct mutation from a background
  thread deadlocks the server, and the symptom is a frozen UI with a still-running
  child process, which is a miserable thing to debug.

If the new family writes a third output convention, extend the dataset scanner's
detection branch and the plot-control construction to match. Do not try to coerce
the new format into one of the existing two — the branch is cheap and the coercion
is where silent indexing bugs come from.
