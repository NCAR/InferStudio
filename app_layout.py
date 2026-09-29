# app_layout.py
import os
import base64
import warnings
import panel as pn
import param
import xarray as xr
from pathlib import Path
from functools import lru_cache

from dimensions import LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME, resolve_nc_glob

from visualization.datasetSelector2 import DatasetBrowser
from visualization.metadata import DatasetMetadata
from visualization.datasetPlot import DatasetPlot2, SharedPlotControls
from visualization.forecastStatsPanel import ForecastStatsPanel
from visualization.loadSuiteDialog import LoadSuiteDialog
from visualization.videoExport import VideoExportPanel
from visualization.plotGrid import PlotGrid, PlotGridState, CLIM_UNSET
from visualization.earth2StudioPlot import close_dataset_cache

from inference.commandRunner import CommandRunner
from inference.inferenceTab import InferenceTab

# Model-difference cache. Each pair gets its own subdirectory (see
# modelDiff.compute_model_difference), so this is the parent only.
DIFF_CACHE_DIR = Path(f"/glade/derecho/scratch/{os.environ['USER']}/.inferstudio_diff_cache")

# --- Static asset locations ------------------------------------------------
# Resolved relative to THIS module, not the process working directory, so the
# paths hold regardless of where `panel serve` is launched from (OOD's
# script.sh.erb does not necessarily cd into the repo root).
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_LOGO_DIR = _STATIC_DIR / "logo"

_MIME = {".png": "image/png", ".ico": "image/x-icon", ".svg": "image/svg+xml"}

@lru_cache(maxsize=None)
def logo_uri(name: str) -> str:
    """Embed a file from static/logo as a data URI (proxy-prefix safe).

    The trailing "#<filename>" fragment exists for Panel: its
    _get_favicon_type() dispatches on the string suffix of the favicon
    value and raises "favicon type not supported" for a bare data URI.
    The fragment satisfies that check and is discarded by the browser
    before the base64 payload is decoded.
    """
    path = _LOGO_DIR / name
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_MIME[path.suffix.lower()]};base64,{data}#{path.name}"

def _resolve_dim(ds, *candidates):
    """Return the first candidate name that exists as a dimension in ds."""
    for name in candidates:
        if name in ds.sizes:
            return name
    return None

def scan_single_dataset(dataset_dir: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with xr.open_mfdataset(resolve_nc_glob(dataset_dir), engine="netcdf4", autoclose=True, data_vars='all') as ds:
            lat_dim = _resolve_dim(ds, LAT_NAME, "lat")
            lon_dim = _resolve_dim(ds, LON_NAME, "lon")

            # Pre-cf_convert earth2studio output has a size-1 `time` dim (the
            # init/cycle time) plus a separate `lead_time` dim holding the
            # actual forecast steps. Post-conversion, cf_convert collapses
            # those into a single `time` axis of VALID times, so this branch
            # only fires for legacy files. ERA5-style files never had
            # lead_time and step through `time` directly, same as converted
            # output does.
            has_lead_time = "lead_time" in ds.sizes and ds.sizes["lead_time"] > 1

            if has_lead_time:
                ntime = int(ds.sizes["lead_time"])
                init_time = ds.time.values[0]
                lead_times = ds.lead_time.values
                stime = str((init_time + lead_times.min()).astype("datetime64[s]"))
                etime = str((init_time + lead_times.max()).astype("datetime64[s]"))
            else:
                ntime = len(ds.time)
                stime = str(ds.time.values[0].astype("datetime64[s]"))
                etime = str(ds.time.values[-1].astype("datetime64[s]"))

            # CF-compliant files (post cf_convert.py) stack pressure levels
            # into a real `pressure` dimension on each leveled variable,
            # rather than earth2studio's original flattened per-level
            # variable names (u100, u850, ...). Capture which variables
            # actually have this real dimension, and the real coordinate
            # values, so the Level (hPa) dropdown can be populated
            # correctly even though these variable names no longer end in
            # a digit (parse_variable_groups can't detect levels from the
            # name alone for these).
            leveled_vars_cf = {}
            if PRES_NAME in ds.coords:
                pressure_values = sorted(float(x) for x in ds[PRES_NAME].values.tolist())
                for v in ds.data_vars:
                    if PRES_NAME in ds[v].dims:
                        leveled_vars_cf[v] = pressure_values

            # Classify by whether a variable actually spans pressure levels,
            # not by raw dimension count. The count test worked only by
            # accident: pre-conversion, surface fields were
            # (time=1, lead_time, lat, lon) = 4 dims, same as leveled ones
            # once you count the flattened name as its own variable - so
            # msl/sp/t2m were landing in vars3d. Post-conversion the counts
            # happen to come out right (3 vs 4), but testing for the
            # pressure dimension says what is actually meant.
            if PRES_NAME in ds.coords:
                vars2d = [v for v in ds.data_vars if PRES_NAME not in ds[v].dims]
                vars3d = [v for v in ds.data_vars if PRES_NAME in ds[v].dims]
            else:
                # Legacy files have no pressure coordinate at all; fall back
                # to the original heuristic for those.
                vars2d = [v for v in ds.data_vars if len(ds[v].dims) <= 3]
                vars3d = [v for v in ds.data_vars if len(ds[v].dims) > 3]

            return {
                "path": str(dataset_dir),
                "ntime": ntime,
                "nlev": len(ds.get(LEV_NAME, [])),
                "nplev": int(ds.sizes.get(PRES_NAME, 0)),
                "nlat": int(ds.sizes[lat_dim]) if lat_dim else 0,
                "nlon": int(ds.sizes[lon_dim]) if lon_dim else 0,
                "stime": stime,
                "etime": etime,
                "vars2d": sorted(vars2d),
                "vars3d": sorted(vars3d),
                "leveled_vars_cf": leveled_vars_cf,
            }

def scan_simulation_suite(sim_dir: Path) -> dict:
    """Scan every model subdirectory under a simulation dir and combine
    them into a single metadata entry representing the whole suite."""
    model_meta = {}
    errors = {}
    for model_dir in sorted(p for p in sim_dir.iterdir() if p.is_dir()):
        try:
            model_meta[model_dir.name] = scan_single_dataset(model_dir)
        except Exception as e:
            errors[model_dir.name] = str(e)

    if not model_meta:
        raise RuntimeError(f"No scannable model outputs found under {sim_dir}")

    any_model = next(iter(model_meta.values()))
    vars2d = sorted(set().union(*(m["vars2d"] for m in model_meta.values())))
    vars3d = sorted(set().union(*(m["vars3d"] for m in model_meta.values())))

    leveled_vars_cf = {}
    for m in model_meta.values():
        for var, levels in m.get("leveled_vars_cf", {}).items():
            leveled_vars_cf.setdefault(var, set()).update(levels)
    leveled_vars_cf = {k: sorted(v) for k, v in leveled_vars_cf.items()}

    return {
        "path": str(sim_dir),
        "models": model_meta,      # per-model breakdown: {model_name: {...}}
        "model_errors": errors,
        "ntime": any_model["ntime"],
        "nlev": any_model["nlev"],
        "nplev": any_model["nplev"],
        "nlat": any_model["nlat"],
        "nlon": any_model["nlon"],
        "stime": any_model["stime"],
        "etime": any_model["etime"],
        "vars2d": vars2d,
        "vars3d": vars3d,
        "leveled_vars_cf": leveled_vars_cf,
    }

def scan_datasets(data_dir):
    metadata = {}
    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        subdirs_with_nc = [
            sub for sub in d.iterdir()
            if sub.is_dir() and any(sub.glob("*.nc"))
        ]
        try:
            if subdirs_with_nc:
                metadata[d.name] = scan_simulation_suite(d)
            else:
                metadata[d.name] = scan_single_dataset(d)
        except Exception as e:
            print(f"Skipping {d.name}: {e}")
            continue
    return metadata


# --- Suite / control adapters ---------------------------------------------
# scan_simulation_suite already produces everything PlotGrid needs; these two
# helpers just reshape it, so the grid never has to know about the metadata
# dict's layout.

def model_dirs_for(entry: dict) -> dict:
    """{model name: directory} for a suite entry, or {} for a flat dataset.

    A flat dataset (ExampleDataset) has no "models" key - it's a single
    directory of .nc files with no per-model breakdown, which is why it
    still routes to DatasetPlot2 below rather than to the model grid.
    """
    return {name: Path(m["path"]) for name, m in entry.get("models", {}).items()}


# Sidebar width shared by the Visualization and Statistics tabs. The busy
# overlay pads its content by the same amount so its spinner lands in the
# middle of the plot card rather than the middle of the whole tab.
_SIDEBAR_WIDTH = 250

def _diff_busy_html(diff_status, diff_pairs):
    """Overlay HTML for the Visualization tab while a difference computes.

    Empty string when nothing is computing. Otherwise a translucent grey
    sheet with a large spinner and one "Computing difference between A and
    B..." line per pair in progress, centred over the plot card.
    """
    lines = [
        f"Computing difference between {model} and {other}\u2026"
        for model, status in diff_status.items()
        if status == "computing" and (other := diff_pairs.get(model))
    ]
    if not lines:
        return ""
    text = "".join(
        f"<div class='diff-busy-text'>{line}</div>" for line in lines)
    return (
        "<style>"
        "@keyframes diff-busy-spin{to{transform:rotate(360deg)}}"
        ".diff-busy{position:absolute;inset:0;display:flex;"
        "flex-direction:column;align-items:center;justify-content:center;"
        f"padding-left:{_SIDEBAR_WIDTH}px;box-sizing:border-box;"
        "background:rgba(233,236,239,0.78);cursor:wait;}"
        ".diff-busy-wheel{width:96px;height:96px;border-radius:50%;"
        "border:10px solid #cfd6de;border-top-color:#007bff;"
        "animation:diff-busy-spin 0.9s linear infinite;margin-bottom:24px;}"
        ".diff-busy-text{font-size:20px;font-weight:600;color:#1f2d3d;"
        "text-align:center;}"
        ".diff-busy-note{margin-top:8px;font-size:14px;color:#4a5866;}"
        "</style>"
        "<div class='diff-busy' role='status' aria-live='polite'>"
        "<div class='diff-busy-wheel'></div>"
        f"{text}"
        "<div class='diff-busy-note'>A full suite can take a minute. The "
        "Statistics and Inference tabs are still available.</div>"
        "</div>"
    )

class TabBusyOverlay(pn.custom.JSComponent):
    """Overlay that also makes the rest of its tab inert while shown.

    Covering the tab only stops the mouse - Tab-key focus and typing still
    reach the widgets underneath. So while `html` is non-empty, the
    component sets the `inert` attribute on its sibling elements (the tab
    content it's laid over), which takes them out of the focus order and
    blocks every kind of input, then clears it when `html` empties.
    Siblings are found through the shadow root because Bokeh renders each
    layout inside its own; the overlay's host sits in the same root as the
    content it covers.
    """

    html = param.String(default="")

    _esm = """
    export function render({ model, el }) {
      const box = document.createElement("div");
      el.appendChild(box);
      function sync() {
        const busy = model.html !== "";
        box.innerHTML = model.html;
        const host = el.getRootNode().host;
        const parent = host && host.parentNode;
        if (!parent) return;
        for (const sib of parent.children) {
          if (sib !== host) sib.inert = busy;
        }
      }
      model.on("html", sync);
      // Layout may attach the host after render; re-sync once it has.
      requestAnimationFrame(sync);
      sync();
    }
    """

class WheelScrollPassthrough(pn.custom.JSComponent):
    """Lets the mouse wheel scroll the page anywhere over a plot except
    its map area.

    BokehJS's WheelZoomTool reports every wheel event over a figure as
    handled - even when the cursor is on an axis, the title, a margin or
    the colorbar, where zoom() does nothing - and Bokeh then cancels the
    event, so the page can't scroll while the cursor is anywhere over a
    plot. With the grid's plots stacked down the page, that left only
    narrow gaps to scroll from.

    One window-level capture listener, installed once per page, runs
    before Bokeh's own. For a wheel over a plot but outside its frame it
    stops the event reaching Bokeh without cancelling it, so the browser
    scrolls as it would over plain page content. Over the frame it does
    nothing and the wheel zooms as before. Being window-level, it covers
    plots created later (a new suite, a re-render) without re-wiring, and
    works when a plot scrolls under a cursor that hasn't moved.

    Plot views are looked up from each document's models and cached by
    their event layer (the element Bokeh listens on); the cache is only
    rebuilt when an event arrives over a layer it hasn't seen.
    """

    _esm = """
    export function render() {
      if (window.__inferstudioWheelPassthrough) { return; }
      window.__inferstudioWheelPassthrough = true;
      let views = new Map();   // events layer element -> plot view
      function rebuild() {
        views = new Map();
        for (const doc of window.Bokeh.documents) {
          for (const m of doc.all_models) {
            if (!("toolbar" in m && "renderers" in m && "left" in m)) { continue; }
            const v = window.Bokeh.index.find_one(m);
            if (v != null && v.canvas_view != null && v.frame != null) {
              views.set(v.canvas_view.events_el, v);
            }
          }
        }
      }
      window.addEventListener("wheel", (e) => {
        const layer = e.composedPath().find(
          (n) => n.classList != null && n.classList.contains("bk-events"));
        if (layer == null) { return; }
        if (!views.has(layer)) { rebuild(); }
        const view = views.get(layer);
        if (view == null) { return; }
        const r = layer.getBoundingClientRect();
        if (!view.frame.bbox.contains(e.clientX - r.left, e.clientY - r.top)) {
          e.stopPropagation();
        }
      }, { capture: true, passive: true });
    }
    """

def link_controls(controls, state):
    """Bridge SharedPlotControls -> PlotGridState.

    SharedPlotControls is the single source of truth; PlotGridState exists
    because the grid's DynamicMap streams need a param object holding
    exactly the fields that should trigger a reload, and no others.

    Note the keys are NOT the same words as the values: the controls object
    predates the grid and uses var_name/level_value/colormap where the grid
    uses variable/level/cmap.
    """
    mapping = {
        "time_index": "time_index",
        "var_name": "variable",
        "level_value": "level",
        "colormap": "cmap",
        "cmap_min": "cmap_min",
        "cmap_max": "cmap_max",
        "boundaries": "boundaries",
    }

    def _coerce(dst, value):
        """Convert a widget value to the type PlotGridState declares.

        Necessary because the widgets yield whatever the widgets yield -
        the Level Select gives a string ("500") while leveled_vars_cf
        stores floats, and param.Integer rejects both. Doing this at the
        boundary keeps the coercion in one place; without it a ValueError
        is raised from inside a param watcher, which propagates out of
        whatever assignment triggered it (typically browser.checked_items)
        rather than showing up anywhere near the real cause.
        """
        if dst == "level":
            # level_value is 0, not None, for surface variables - see
            # SharedPlotControls._update_level_options. Passing 0 through
            # would make load_e2s_field try to select pressure=0.
            if value in (None, "", "None", 0):
                return None
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None
        if dst == "time_index":
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
        if dst == "cmap":
            # cmocean's colormaps are not registered with matplotlib
            # globally, so a bare name string can fail downstream. Resolve
            # through the controls' own name->Colormap dict, which is what
            # DatasetPlot2 does too.
            cm = getattr(controls, "_colormaps", {}).get(value)
            if cm is not None:
                return cm
            return value if isinstance(value, str) else "viridis"
        if dst in ("cmap_min", "cmap_max"):
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return value

    for src, dst in mapping.items():
        if not hasattr(controls, src):
            print(f"link_controls: SharedPlotControls has no {src!r} - skipping")
            continue
        # Push the current value across before wiring the watcher, so the
        # grid's first render already reflects the sidebar rather than the
        # PlotGridState defaults.
        setattr(state, dst, _coerce(dst, getattr(controls, src)))
        controls.param.watch(
            lambda ev, dst=dst: setattr(state, dst, _coerce(dst, ev.new)), src
        )


def build_app(data_dir):
    dataset_metadata = scan_datasets(data_dir)
    datasets = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    browser = DatasetBrowser(datasets=datasets)
    meta_panel = DatasetMetadata(metadata=dataset_metadata)

    # The Statistics tab gets its own dataset browser widget (a Panel
    # component can't be embedded in two tabs' layouts at once), mirrored
    # two-way with `browser` so dataset selection is genuinely shared
    # between the Visualization and Statistics tabs rather than two
    # independent selections.
    stats_browser = DatasetBrowser(datasets=datasets)

    def _mirror(dst, attr):
        def _watcher(event):
            if getattr(dst, attr) != event.new:
                setattr(dst, attr, event.new)
        return _watcher

    browser.param.watch(_mirror(stats_browser, "checked_items"), "checked_items")
    browser.param.watch(_mirror(stats_browser, "active_dataset"), "active_dataset")
    stats_browser.param.watch(_mirror(browser, "checked_items"), "checked_items")
    stats_browser.param.watch(_mirror(browser, "active_dataset"), "active_dataset")

    controls = SharedPlotControls()
    controls.update_choices(browser.checked_items, dataset_metadata)

    # One PlotGridState for the whole session, outliving every PlotGrid built
    # against it. This is what lets the suite change without rebinding the
    # sidebar: binding watchers to a grid instance instead would leave a
    # stale watcher attached on every dataset switch, and they'd accumulate.
    grid_state = PlotGridState()
    link_controls(controls, grid_state)

    def _reflect_field_clim(event):
        """Push the suite grid's colour range back onto the sidebar's
        Min/Max boxes.

        link_controls only wires controls -> state, so PlotGrid's own
        auto-scaling (refresh_clims, run when the variable/level/dataset
        changes) and colorbar drags never reach the boxes the other
        direction - they'd sit at their initial 0.0 forever regardless of
        what the grid is actually plotting.

        Display-only (_set_displayed_*), so this never marks the range as
        user-fixed. It used to be skipped once the user had typed an
        explicit value, but a colorbar drag can now move the range away
        from that value, and the boxes should show what's on screen.
        """
        lo, hi = event.new
        if (lo, hi) == CLIM_UNSET:
            return
        controls._set_displayed_min(lo)
        controls._set_displayed_max(hi)

    grid_state.param.watch(_reflect_field_clim, "field_clim")

    def sync_active(event):
        meta_panel.active_key = event.new

    browser.param.watch(sync_active, 'active_dataset')

    def sync_controls(event):
        controls.update_choices(event.new, dataset_metadata)

    browser.param.watch(sync_controls, 'checked_items')

    DEFAULT_DATASET = "ExampleDataset"
    if DEFAULT_DATASET in dataset_metadata:
        browser.checked_items = [DEFAULT_DATASET]
        browser.active_dataset = DEFAULT_DATASET
    elif DEFAULT_DATASET and DEFAULT_DATASET != "REPLACE_WITH_YOUR_FOLDER_NAME":
        print(f"Warning: default dataset {DEFAULT_DATASET!r} not found under {data_dir}")

    inference_tab = InferenceTab()

    def _on_new_output(event):
        sim_dir = Path(event.new)
        if not sim_dir.is_dir():
            if pn.state.notifications:
                pn.state.notifications.error(f"sim_dir not found: {sim_dir}", duration=0)
            return

        key = sim_dir.name
        try:
            dataset_metadata[key] = scan_simulation_suite(sim_dir)
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Could not scan {key}: {e}", duration=0)
            with open('/tmp/debug.log', 'a') as f:
                f.write(f"scan_simulation_suite failed for {key}: {e}\n")
            return
        for model, err in dataset_metadata[key].get("model_errors", {}).items():
            if pn.state.notifications:
                pn.state.notifications.error(f"Could not scan {key}/{model}: {err}", duration=0)

        try:
            meta_panel.metadata = dict(dataset_metadata)
            browser.add_datasets([key])
            stats_browser.add_datasets([key])
            if browser.checked_items != [key]:
                browser.checked_items = [key]
            browser.active_dataset = key
            tabs.active = 0
            if pn.state.notifications:
                pn.state.notifications.success(
                    f"Simulation suite '{key}' has finished and was added "
                    "to the Visualization tab.",
                    duration=0,
                )
        except Exception as e:
            # A failure anywhere in here (dataset-browser wiring, the plot
            # grid rebuild it triggers, ...) must not propagate back out of
            # this watcher: it's invoked synchronously from the Inference
            # tab's own completion callback, and an uncaught exception here
            # would abort that callback partway through, leaving the Run/
            # Cancel buttons and spinner stuck as if the run were still going.
            if pn.state.notifications:
                pn.state.notifications.error(
                    f"Scanned {key}, but couldn't load it into the "
                    f"Visualization tab: {e}", duration=0,
                )

    inference_tab.param.watch(_on_new_output, 'outputDirectory')

    # --- Load Existing Suite (Visualization tab) ---------------------- #
    # Lets a user browse to and select a simulation suite directory from
    # a PREVIOUS InferStudio session (rather than only ever seeing suites
    # produced in the current session), reusing the exact same
    # scan_simulation_suite/AIFS-Aurora-etc. scanning logic used for
    # freshly-completed inference runs above.
    def _scan_and_register_suite(path_str, dialog):
        """Shared scan+register logic for a "Load Existing Suite" dialog on
        either the Visualization or the Statistics tab. Both tabs' dataset
        browsers mirror each other (see _mirror above), so registering the
        new suite against `browser` here is enough for it to also appear,
        selected, on whichever tab's dialog wasn't used."""
        sim_dir = Path(path_str)

        if not sim_dir.is_dir():
            dialog.report_error(f"{sim_dir} is not a directory.")
            return

        key = sim_dir.name
        try:
            dataset_metadata[key] = scan_simulation_suite(sim_dir)
        except Exception as e:
            # scan_simulation_suite raises RuntimeError specifically when
            # no supported model (AIFS, Aurora, ...) output was found -
            # this is also where any other scan failure surfaces (e.g.
            # unreadable/corrupt files). Full detail (including the full
            # path) goes in the dialog's own inline error; the toast
            # notification is kept short deliberately, since Notyf-style
            # toasts have a fixed size and don't wrap/expand for long
            # text - putting the full path + exception text in the toast
            # was getting visually clipped.
            dialog.report_error(
                f"Could not load a simulation suite from {sim_dir}: {e}"
            )
            if pn.state.notifications:
                pn.state.notifications.error(
                    f"Could not load suite from {key} \u2014 see dialog for details.",
                    duration=0,
                )
            return

        for model, err in dataset_metadata[key].get("model_errors", {}).items():
            if pn.state.notifications:
                pn.state.notifications.error(f"Could not scan {key}/{model}: {err}", duration=0)

        meta_panel.metadata = dict(dataset_metadata)
        browser.add_datasets([key])
        stats_browser.add_datasets([key])
        if browser.checked_items != [key]:
            browser.checked_items = [key]
        browser.active_dataset = key
        dialog.close()
        if pn.state.notifications:
            pn.state.notifications.info(f"Loaded suite: {key}", duration=0)

    load_suite_dialog = LoadSuiteDialog(
        start_path=Path(f"/glade/derecho/scratch/{os.environ['USER']}"),
        on_select=lambda path_str: _scan_and_register_suite(path_str, load_suite_dialog),
    )
    # Match the Datasets checkbox panel's width exactly: that panel is
    # sizing_mode="stretch_width" with margin=(0, 10, 0, 0) (a 10px right
    # margin - see DatasetBrowser._column in datasetSelector2.py), so
    # giving this button the identical sizing_mode + right margin makes
    # both stretch to the exact same effective width within the sidebar.
    load_suite_dialog.open_button.sizing_mode = "stretch_width"
    load_suite_dialog.open_button.margin = (10, 10, 0, 0)

    stats_load_suite_dialog = LoadSuiteDialog(
        start_path=Path(f"/glade/derecho/scratch/{os.environ['USER']}"),
        on_select=lambda path_str: _scan_and_register_suite(path_str, stats_load_suite_dialog),
    )
    stats_load_suite_dialog.open_button.sizing_mode = "stretch_width"
    stats_load_suite_dialog.open_button.margin = (10, 10, 0, 0)

    # Holds whatever is currently on screen - a PlotGrid for a simulation
    # suite, or a DatasetPlot2 for a flat single dataset. A dict rather than
    # a bare local because plot_grid (a closure) rebinds it on every dataset
    # change, and the video exporter needs to see the new value.
    _active_plot = {"obj": None}

    # Sidebar container for the per-model difference selectors. These used to
    # sit under each model's plot card; hv.Layout can't host Panel widgets
    # between its panels, so they live here alongside the other field
    # controls. Repopulated by plot_grid on every dataset change.
    diff_slot = pn.Column(sizing_mode="stretch_width", margin=(0, 10, 0, 0))

    @pn.depends(browser.param.checked_items)
    def plot_grid(datasets):
        # Detach the outgoing grid before building its replacement. Its
        # streams stay subscribed to the shared grid_state otherwise, so it
        # would keep loading fields from the previous suite on every slider
        # tick and race the new grid over the readout.
        prev = _active_plot.get("obj")
        if hasattr(prev, "teardown"):
            prev.teardown()

        if not datasets:
            _active_plot["obj"] = None
            diff_slot.objects = []
            return pn.pane.Markdown("### Select one or more datasets")

        ds = datasets[0]
        entry = dataset_metadata.get(ds, {})
        model_dirs = model_dirs_for(entry)

        if not model_dirs:
            # Flat dataset (no per-model subdirectories) - there are no model
            # pairs to difference and no grid to link, so this keeps the
            # original single-plot path.
            plot = DatasetPlot2(controls=controls, dataset=ds, metadata=dataset_metadata)
            _active_plot["obj"] = plot
            diff_slot.objects = []
            return plot.panel()

        grid = PlotGrid(
            models=list(model_dirs),
            model_dirs=model_dirs,
            # Scoped per suite (ds is the suite's own directory name) --
            # otherwise a diff cached under "<A>_minus_<B>" for one suite
            # gets silently reused for any other suite that happens to
            # reuse the same model names, comparing two unrelated runs
            # against each other.
            diff_cache_dir=DIFF_CACHE_DIR / ds,
            state=grid_state,
        )
        # ntime already came out of the scan, so the forecast length is set
        # without touching the filesystem again.
        grid.set_time_bounds(entry.get("ntime", 1))
        _active_plot["obj"] = grid
        diff_slot.objects = [grid.diff_selectors()]
        grid.refresh_clims_async()
        return grid.card(title=ds)

    @pn.depends(browser.param.checked_items)
    def stats_reactive(datasets):
        if not datasets:
            return pn.pane.Markdown("### Select one or more datasets")
        ds = datasets[0]
        stats = ForecastStatsPanel(
            controls=controls, dataset_key=ds, metadata=dataset_metadata)
        return stats.panel()

    # Export Video - sweeps the shared time index across the full forecast
    # and encodes the current rendering to MP4 with ffmpeg.
    #
    # NOTE: with the grid now rendered by Bokeh client-side, there are no
    # server-side image buffers to capture. VideoExportPanel must render its
    # own frames via earth2StudioPlot.plot_e2s_field (retained for exactly
    # this reason) using the spec returned by PlotGrid.frame_spec().
    video_export = VideoExportPanel(controls, lambda: _active_plot["obj"])

    sidebar = pn.Column(
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Datasets</h2>"),
        browser.panel,
        load_suite_dialog.open_button,
        load_suite_dialog.modal,
        controls.panel(),
        diff_slot,
        video_export.open_button,
        video_export.modal,
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Metadata</h2>"),
        meta_panel.panel,
        width=_SIDEBAR_WIDTH,
        # Its own bounded scrollbar too - same reasoning as vis/main below:
        # sizing_mode stays default (Bokeh doesn't fight the height here
        # since nothing asks it to manage that axis), height:100% resolves
        # against vis's real box, and this is what keeps a long dataset
        # list from pushing vis's own content past its box (which vis's
        # overflow:hidden would otherwise just clip invisibly instead of
        # making reachable via a scrollbar).
        styles={"height": "100%", "overflow-y": "auto"},
    )
    main = pn.Column(
        pn.panel(plot_grid, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        css_classes=["main-content"],
        # height/overflow set here directly, not just via the .main-content
        # class: Bokeh 3 renders each layout model (this Column included)
        # inside its own shadow root, and raw_css (how static/styles.css
        # gets loaded) is injected as one global <style> tag in the page's
        # light-DOM <head> - it never crosses into a shadow root, so
        # .main-content's height/overflow rules were silently inert here
        # (confirmed empirically: computed overflow-y stayed "visible").
        # `styles=` instead sets a genuine inline style on this element's
        # own node, immune to that boundary - same fix already applied to
        # vis/inference/sidebar/tabs above.
        #
        # overflow-x too: the plot grid is fixed-size (see PlotGrid), so a
        # narrow window scrolls it sideways here instead of clipping it.
        styles={"height": "100%", "overflow-y": "auto", "overflow-x": "auto"},
    )
    # height:100% here, NOT 100vh - and sizing_mode="stretch_width" NOT
    # stretch_both/stretch_height. That second part matters as much as the
    # first: "stretch_both" hands height control to Bokeh's own JS layout
    # solver, which computes each LayoutDOM's height from its CHILDREN's
    # natural content size (it has no way to see #main's actual box, which
    # is a plain templated div, not something Bokeh's resize machinery
    # tracks) - measured via Playwright, a stretch_both wrapper here came
    # out at a JS-computed 1260px tall against an 810px-tall #main, because
    # Bokeh's layout pass sets an explicit inline `style.height` on every
    # pass, silently clobbering whatever percentage we'd set via `styles=`
    # on that same element. stretch_width leaves the height axis alone, so
    # our own CSS height:100% is what actually applies - the same pattern
    # `.main-content` (below) already used successfully.
    #
    # With that, #main's real box (see bootstrap.html/bootstrap.css:
    # #container is vh-100 + overflow-hidden, #content and #main are
    # height:100%, #main additionally has its own overflow-y:auto) finally
    # reaches these panes correctly: 100% now resolves to #main's true
    # height instead of an oversized intrinsic one, so the tab bar (a
    # sibling within that same non-scrolling chain) never gets carried off
    # by an internal scroll, and each pane's own overflow-y:auto is what
    # scrolls - not #main, and not the page.
    vis_content = pn.Row(sidebar, main, sizing_mode="stretch_width",
                         styles={"height": "100%", "overflow": "hidden"})

    # Greys out and locks the whole Visualization tab while a difference
    # computes. The small header spinner was the only other sign, and it was
    # easy to miss; the old in-card spinner still left the sidebar live, so
    # the user could change the variable or pick another pair mid-run.
    # Absolutely positioned over vis_content rather than swapped in for it,
    # so the tab keeps its layout underneath and nothing reflows when the
    # computation finishes. Lives inside the tab, so Statistics and
    # Inference stay usable.
    diff_busy_overlay = TabBusyOverlay(
        visible=False,
        margin=0,
        styles={
            "position": "absolute", "top": "0", "left": "0",
            "width": "100%", "height": "100%", "z-index": "1000",
        },
    )

    def _sync_diff_busy(*_):
        html = _diff_busy_html(grid_state.diff_status, grid_state.diff_pairs)
        diff_busy_overlay.html = html
        diff_busy_overlay.visible = bool(html)

    grid_state.param.watch(_sync_diff_busy, ["diff_status", "diff_pairs"])

    vis = pn.Column(
        vis_content, diff_busy_overlay,
        WheelScrollPassthrough(width=0, height=0, margin=0),
        sizing_mode="stretch_width",
        styles={"height": "100%", "overflow": "hidden",
                "position": "relative"},
    )
    inference = pn.Column(
        inference_tab.panel(),
        sizing_mode="stretch_width",
        styles={"height": "100%", "overflow-y": "auto"},
    )

    statistics_sidebar = pn.Column(
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Datasets</h2>"),
        stats_browser.panel,
        stats_load_suite_dialog.open_button,
        stats_load_suite_dialog.modal,
        width=_SIDEBAR_WIDTH,
        styles={"height": "100%", "overflow-y": "auto"},
    )
    statistics_main = pn.Column(
        pn.panel(stats_reactive, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        css_classes=["main-content"],
        styles={"height": "100%", "overflow-y": "auto"},
    )
    statistics = pn.Row(
        statistics_sidebar, statistics_main, sizing_mode="stretch_width",
        styles={"height": "100%", "overflow": "hidden"},
    )

    tabs = pn.Tabs(
        ("Visualization", vis),
        ("Statistics", statistics),
        ("Inference", inference),
        # stretch_width, not stretch_both - see the comment above. Height
        # comes from the .bk-tabs-content rule below instead.
        sizing_mode="stretch_width",
        styles={"height": "100%", "overflow": "hidden"},
        stylesheets=["""
            .bk-tab {
                background: #f0f0f0; border-radius: 4px 4px 0 0;
                font-size: 18px; font-weight: 600; padding: 12px 24px;
            }
            .bk-tab.bk-active { background: white; border-top: 3px solid #007bff; font-weight: 700; }
            .bk-tabs-header { background: #e8e8e8; flex: 0 0 auto; }
            .bk-tabs-content {
                border: 1px solid #ccc; padding: 10px; box-sizing: border-box;
                flex: 1 1 auto; min-height: 0; overflow: hidden;
            }
        """],
    )
    # `title` now only drives the browser tab text - the header title text is
    # supplied by the wordmark image below, so it is no longer set to "".
    template = pn.template.BootstrapTemplate(
        title="InferStudio",
        favicon=logo_uri("favicon.ico"),
        header_background="#091422",
        busy_indicator=None,
    )
    # InferStudio wordmark, replacing the plain-text HTML title pane. The
    # HSpacer takes over the layout job the old pane's stretch_width was
    # doing - pushing the spinner and NSF NCAR logo to the right edge.
    template.header.append(
        pn.pane.PNG(
            str(_LOGO_DIR / "wordmark_dark.png"),
            height=64,
            width=161,
            sizing_mode="fixed",
            margin=(5, 0, 5, 10),
        )
    )
    template.header.append(pn.layout.HSpacer())
    busy_spinner = pn.indicators.LoadingSpinner(
        value=False, width=20, height=20, color="light",
        margin=(10, 10, 10, 0),
    )
    # Documentation and GitHub Discussions links. Both live in one pane with
    # an explicit width: Panel sizes an HTML pane's box independently of its
    # text, so separate panes let the longer label spill over its
    # neighbours instead of pushing them aside.
    _link_style = (
        'style="display:inline-flex;align-items:center;gap:7px;'
        'color:#DFEFF6;font-size:14px;font-weight:500;'
        'text-decoration:none;white-space:nowrap;"'
    )
    _icon_attrs = (
        'width="16" height="16" viewBox="0 0 16 16" fill="none" '
        'stroke="currentColor" stroke-width="1.4" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true"'
    )
    template.header.append(
        pn.pane.HTML(
            '<div style="display:flex;align-items:center;'
            'justify-content:flex-end;gap:24px;">'
            '<a href="https://inferstudio.readthedocs.io/" target="_blank" '
            f'rel="noopener" title="InferStudio documentation" {_link_style}>'
            f'<svg {_icon_attrs}>'
            '<path d="M2 2.5h4a2 2 0 0 1 2 2v9a1.5 1.5 0 0 0-1.5-1.5H2z"/>'
            '<path d="M14 2.5h-4a2 2 0 0 0-2 2v9a1.5 1.5 0 0 1 1.5-1.5H14z"/>'
            '</svg>Docs</a>'
            '<a href="https://github.com/NCAR/InferStudio/discussions" '
            'target="_blank" rel="noopener" '
            'title="Ask questions and share feedback on GitHub Discussions" '
            f'{_link_style}>'
            f'<svg {_icon_attrs}>'
            '<path d="M2.5 3h11a1 1 0 0 1 1 1v6.5a1 1 0 0 1-1 1H7l-3 2.5v-2.5'
            'H2.5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/>'
            '</svg>Questions &amp; Feedback</a>'
            '</div>',
            width=280,
            sizing_mode="fixed",
            # The header row is a non-wrapping flex row whose items all
            # default to flex-shrink:1, so on narrower windows the browser
            # squeezed this pane below its content and the labels ran under
            # the spinner and logos. Pin it at its full width.
            styles={
                "display": "flex", "align-items": "center", "height": "100%",
                "flex-shrink": "0", "min-width": "280px",
            },
            margin=(0, 40, 0, 0),
        )
    )
    pn.state.sync_busy(busy_spinner)
    template.header.append(busy_spinner)
    template.header.append(
        pn.pane.PNG(
            str(_STATIC_DIR / "nsf_ncar_logo_padded.png"),
            height=45,
            width=534,
            sizing_mode="fixed",
            margin=(5, 0, 5, 0),
        )
    )
    # stretch_width + explicit CSS height:100%, not stretch_both - see the
    # comment above vis/inference/tabs for why stretch_both silently
    # breaks this exact chain.
    template.main[:] = [pn.Column(
        tabs, sizing_mode="stretch_width",
        styles={"height": "100%", "overflow": "hidden"},
    )]

    # Deferred via pn.state.onload rather than called directly here:
    # pn.state.notifications requires the browser session to be fully
    # connected before it can actually display anything client-side.
    # Calling .info(...) synchronously at this point in build_app() runs
    # before that connection is guaranteed to be live, so the message was
    # being silently dropped - this is why the welcome message never
    # appeared on initial launch, while the (unrelated) notification fired
    # from _on_new_output above worked fine, since by the time an
    # inference run completes the session has obviously been live for a
    # while already.
    def _show_welcome():
        if pn.state.notifications:
            pn.state.notifications.info(
                "Welcome to InferStudio.<br><br>"
                "You are currently viewing information from "
                "an example dataset. To run your own AI weather model inference, go "
                "to the Inference tab.<br><br> Then select your desired parameters, click "
                "\"Run Inference,\" and your simulation suite will be viewable from here."
                "<br><br><br>",
                duration=0,
            )

    pn.state.onload(_show_welcome)

    # earth2StudioPlot holds datasets open (dask-backed) so the grid's
    # DynamicMap callbacks don't reopen an mfdataset on every slider tick.
    # Release the file handles when the session ends - on a long-lived OOD
    # server these would otherwise accumulate across sessions.
    pn.state.on_session_destroyed(lambda ctx: close_dataset_cache())

    return template

# To drop jupyter in the future:
if __name__ == "__main__": build_app(DATA_DIR).servable()
