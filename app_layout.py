# app_layout.py
import warnings
import xarray as xr
import panel as pn
from pathlib import Path
from datasetSelector2 import DatasetBrowser
from metadata import DatasetMetadata
from datasetPlot import DatasetPlot2
from commandRunner import CommandRunner
from inferenceTab import InferenceTab
from era5_plot import LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME


def scan_datasets(data_dir):
    metadata = {}
    for d in data_dir.iterdir():
        if d.is_dir():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
            with xr.open_mfdataset(f"{d}/*.nc", engine="netcdf4", autoclose=True, data_vars='all') as ds:
                metadata[d.name] = {
                    "ntime": len(ds.time),
                    "nlev": len(ds.get(LEV_NAME, [])),
                    "nplev": int(ds.sizes[PRES_NAME]),
                    "nlat": int(ds.sizes[LAT_NAME]),
                    "nlon": int(ds.sizes[LON_NAME]),
                    "stime": str(ds.time.values[0].astype("datetime64[s]")),
                    "etime": str(ds.time.values[-1].astype("datetime64[s]")),
                    "vars2d": [v for v in ds.data_vars if len(ds[v].dims) <= 3],
                    "vars3d": [v for v in ds.data_vars if len(ds[v].dims) > 3],
                }
    return metadata


def build_app(data_dir):
    dataset_metadata = scan_datasets(data_dir)
    datasets = sorted(d.name for d in data_dir.iterdir() if d.is_dir())

    browser = DatasetBrowser(datasets=datasets)
    meta_panel = DatasetMetadata(metadata=dataset_metadata)

    def sync_active(event):
        meta_panel.active_key = event.new
    browser.param.watch(sync_active, 'active_dataset')

    @pn.depends(browser.param.checked_items)
    def plot_grid(datasets):
        if not datasets:
            return pn.pane.Markdown("### Select one or more datasets")
        plots = [DatasetPlot2(dataset=ds, metadata=dataset_metadata).panel() for ds in datasets]
        return pn.GridBox(*plots, ncols=2, sizing_mode=None, css_classes=["plot-grid"],
                          styles={"grid-auto-rows": "min-content", "align-items": "start"})

    sidebar = pn.Column(
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Datasets</h2>"),
        browser.panel,
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Metadata</h2>"),
        meta_panel.panel,
        width=250,
    )
    main = pn.Column(plot_grid, sizing_mode="stretch_width", css_classes=["main-content"])
    vis = pn.Row(sidebar, main, sizing_mode="stretch_both", styles={"height": "100vh"})

    inference = InferenceTab().panel()

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

    template = pn.template.BootstrapTemplate(title="InferStudio")
    template.main[:] = [pn.Column(tabs, sizing_mode="stretch_both")]
    return template

# To drop jupyter in the future:
# if __name__ == "__main__": build_app(DATA_DIR).servable()
