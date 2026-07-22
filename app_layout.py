# app_layout.py
import warnings
import xarray as xr
import panel as pn
from pathlib import Path
from datasetSelector2 import DatasetBrowser
from metadata import DatasetMetadata
from datasetPlot import DatasetPlot2, SharedPlotControls
from forecastStatsPanel import ForecastStatsPanel
from commandRunner import CommandRunner
from inferenceTab import InferenceTab
from era5_plot import LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME

def _resolve_dim(ds, *candidates):
    """Return the first candidate name that exists as a dimension in ds."""
    for name in candidates:
        if name in ds.sizes:
            return name
    return None

def scan_single_dataset(dataset_dir: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with xr.open_mfdataset(f"{dataset_dir}/*.nc", engine="netcdf4", autoclose=True, data_vars='all') as ds:
            lat_dim = _resolve_dim(ds, LAT_NAME, "lat")
            lon_dim = _resolve_dim(ds, LON_NAME, "lon")
            return {
                "path": str(dataset_dir),
                "ntime": len(ds.time),
                "nlev": len(ds.get(LEV_NAME, [])),
                "nplev": int(ds.sizes.get(PRES_NAME, 0)),
                "nlat": int(ds.sizes[lat_dim]) if lat_dim else 0,
                "nlon": int(ds.sizes[lon_dim]) if lon_dim else 0,
                "stime": str(ds.time.values[0].astype("datetime64[s]")),
                "etime": str(ds.time.values[-1].astype("datetime64[s]")),
                "vars2d": [v for v in ds.data_vars if len(ds[v].dims) <= 3],
                "vars3d": [v for v in ds.data_vars if len(ds[v].dims) > 3],
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
    }

def scan_datasets(data_dir):
    metadata = {}
    for d in data_dir.iterdir():
        if d.is_dir():
            try:
                metadata[d.name] = scan_single_dataset(d)
            except Exception as e:
                print(f"Skipping {d.name}: {e}")
                continue
    return metadata


def build_app(data_dir):
    dataset_metadata = scan_datasets(data_dir)
    datasets = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    browser = DatasetBrowser(datasets=datasets)
    meta_panel = DatasetMetadata(metadata=dataset_metadata)

    # Shared variable/level controls — one set drives every plot in the
    # Visualization tab, rather than each DatasetPlot2 owning its own.
    controls = SharedPlotControls()
    controls.update_choices(browser.checked_items, dataset_metadata)

    def sync_active(event):
        meta_panel.active_key = event.new

    browser.param.watch(sync_active, 'active_dataset')

    def sync_controls(event):
        controls.update_choices(event.new, dataset_metadata)

    browser.param.watch(sync_controls, 'checked_items')

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
        with open('/tmp/debug.log', 'a') as f:
            f.write(f"scanned {key}: vars2d={dataset_metadata[key]['vars2d']} vars3d={dataset_metadata[key]['vars3d']} models={list(dataset_metadata[key].get('models', {}).keys())}\n")
        for model, err in dataset_metadata[key].get("model_errors", {}).items():
            if pn.state.notifications:
                pn.state.notifications.error(f"Could not scan {key}/{model}: {err}", duration=0)
        meta_panel.metadata = dict(dataset_metadata)
        browser.add_datasets([key])
        if browser.checked_items != [key]:
            browser.checked_items = [key]
        browser.active_dataset = key
        tabs.active = 0
        if pn.state.notifications:
            pn.state.notifications.info(f"browser now has: {browser.datasets}", duration=0)

    inference_tab.param.watch(_on_new_output, 'outputDirectory')

    @pn.depends(browser.param.checked_items)
    def plot_grid(datasets):
        if not datasets:
            return pn.pane.Markdown("### Select one or more datasets")
        # Only one dataset can be checked at a time (see DatasetBrowser),
        # so just render it directly rather than wrapping it in a GridBox —
        # GridBox's column sizing doesn't reliably stretch to fill available
        # width even with sizing_mode="stretch_width", whereas a plain
        # stretch_width Column (as used elsewhere in this app) does.
        ds = datasets[0]
        return DatasetPlot2(controls=controls, dataset=ds, metadata=dataset_metadata).panel()

    @pn.depends(browser.param.checked_items)
    def stats_panel(datasets):
        if not datasets:
            return pn.pane.Markdown("")
        ds = datasets[0]
        return ForecastStatsPanel(controls=controls, dataset_key=ds, metadata=dataset_metadata).panel()

    sidebar = pn.Column(
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Datasets</h2>"),
        browser.panel,
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Variable &amp; Level</h2>"),
        controls.panel(),
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Metadata</h2>"),
        meta_panel.panel,
        width=250,
    )
    main = pn.Column(
        pn.panel(plot_grid, sizing_mode="stretch_width"),
        pn.panel(stats_panel, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        css_classes=["main-content"],
    )
    vis = pn.Row(sidebar, main, sizing_mode="stretch_both", styles={"height": "100vh"})
    inference = pn.Column(
        inference_tab.panel(),
        sizing_mode="stretch_both",
        styles={"height": "100vh", "overflow": "auto"},
    )
    tabs = pn.Tabs(
        ("Visualization", vis),
        ("Inference", inference),
        stylesheets=["""
            .bk-tab { background: #f0f0f0; border-radius: 4px 4px 0 0; font-size: 14px; padding: 8px 16px; }
            .bk-tab.bk-active { background: white; border-top: 2px solid #007bff; font-weight: bold; }
            .bk-tabs-header { background: #e8e8e8; }
            .bk-tabs-content { border: 1px solid #ccc; padding: 10px; }
        """],
    )
    template = pn.template.BootstrapTemplate(title="", busy_indicator=None)
    template.header.append(
        pn.pane.HTML(
            "InferStudio",
            styles={
                "font-size": "20px",
                "font-weight": "600",
                "color": "white",
                "white-space": "nowrap",
                "overflow": "hidden",
                "text-overflow": "ellipsis",
                "min-width": "0",
                "padding-left": "10px",
            },
            sizing_mode="stretch_width",
        )
    )
    busy_spinner = pn.indicators.LoadingSpinner(
        value=False, width=20, height=20, color="light",
        margin=(10, 10, 10, 0),
    )
    pn.state.sync_busy(busy_spinner)  # re-wire actual busy-state reflection, since we disabled the built-in one
    template.header.append(busy_spinner)
    template.header.append(
        pn.pane.PNG(
            "static/nsf_ncar_logo_padded.png",
            height=45,
            width=534,
            sizing_mode="fixed",
            margin=(5, 0, 5, 0),
        )
    )
    template.main[:] = [pn.Column(tabs, sizing_mode="stretch_both")]
    return template

# To drop jupyter in the future:
# if __name__ == "__main__": build_app(DATA_DIR).servable()
