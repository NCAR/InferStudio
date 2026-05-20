#!/usr/bin/env python
# coding: utf-8

# Panel App – How to Run on NCAR JupyterHub (HPC)
# 
# 1) Log into JupyterHub:
#    https://jupyterhub.hpc.ucar.edu/
# 
# 2) Open a Terminal in JupyterLab
# 
# 3) Activate the conda environment (if needed):
#    conda activate <YOUR_CONDA_ENV>
# 
# 4) Navigate to this file’s directory
# 
# 5) Start the Panel server (keep this terminal running):
#    panel serve panel_app.py --address 127.0.0.1 --port 5006 \
#        --allow-websocket-origin="jupyterhub.hpc.ucar.edu"
# 
# 6) Open the app in your browser (same JupyterHub session):
#    https://jupyterhub.hpc.ucar.edu/stable/user/<USER_NAME>/proxy/5006/panel_app
# 
# 7) Stop the app:
#    Go back to the terminal and press Ctrl+C

# In[1]:


import xarray as xr
from pathlib import Path
from era5_plot import plot_png, NETCDF_FILE, VAR_NAME, TIME_NAME, LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME
import panel as pn
import param


# In[2]:


from datasetSelector2 import DatasetBrowser
from metadata import DatasetMetadata
from datasetPlot import DatasetPlot2
from commandRunner import CommandRunner
from inferenceTab import InferenceTab


# In[3]:


#pn.extension(raw_css=[Path("static/styles.css").read_text()])
pn.extension('modal', 
             raw_css=[".bk-btn-group { flex-wrap: wrap !important; max-width: 600px; }",
                      ".bk-btn-group button { border-radius: 4px !important; margin: 2px;}"
                     ]
            )


# In[4]:


DATA_DIR = Path("/Users/vapor/Data/model_predict")
#DATA_DIR = Path("/glade/derecho/scratch/pearse/CREDIT/RAW_OUTPUT/panelTest/")
DATASET_METADATA = {}


# In[5]:


def scan_datasets():
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            nc_file = f"{d}/*.nc"
            with xr.open_mfdataset(nc_file, engine="netcdf4", autoclose=True) as ds:
                DATASET_METADATA[d.name] = {
                    "ntime": len(ds.time),
                    "nlev": len(ds.get(LEV_NAME, [])),
                    "nplev": int(ds.sizes[PRES_NAME]),
                    "nlat": int(ds.sizes[LAT_NAME]),
                    "nlon": int(ds.sizes[LON_NAME]),
                    "stime": str(ds.time.values[0].astype("datetime64[s]")),
                    "etime": str(ds.time.values[-1].astype("datetime64[s]")),
                    "vars2d": [v for v in ds.data_vars if len(ds[v].dims) <= 3],
                    "vars3d": [v for v in ds.data_vars if len(ds[v].dims) > 3]
                }


# In[6]:


scan_datasets()


# In[7]:


def available_datasets():
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir()
    )


# In[8]:


browser = DatasetBrowser(datasets=available_datasets())


# In[9]:


@pn.depends(browser.param.checked_items)
def plot_grid(datasets):

    if not datasets:
        return pn.pane.Markdown("### Select one or more datasets")

    plots = [
        DatasetPlot2(dataset=ds, metadata=DATASET_METADATA).panel()
        for ds in datasets
    ]

    return pn.GridBox(
        *plots,
        ncols=2,
        sizing_mode=None,
        css_classes=["plot-grid"],
        styles={
            "grid-auto-rows": "min-content",
            "align-items": "start"
        },
    )


# In[10]:


metadata = DatasetMetadata(metadata=DATASET_METADATA)
def sync_active_dataset(event):
    metadata.active_key = event.new
browser.param.watch(sync_active_dataset, 'active_dataset')


# In[11]:


sidebar = pn.Column(
    #"## Datasets",
    pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Datasets</h2>"),
    browser.panel,
    pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Metadata</h2>"),
    metadata.panel,
    width=250
)


# In[12]:


main = pn.Column(
    plot_grid,
    sizing_mode="stretch_width",
    css_classes=["main-content"]    
)


# In[13]:


vis = pn.Row(
    sidebar,
    main,
    sizing_mode="stretch_both",
    styles={"height" : "100vh"}
)


# In[14]:


template = pn.template.BootstrapTemplate(title="Forecast Studio")


# In[15]:


commandRunner = CommandRunner()
inference = pn.Column(
    commandRunner.panel()
)


# In[16]:


inferenceTab = InferenceTab()

tabs = pn.Tabs(
    ("Visualization", vis),
    #("Inference", inference),
    ("Inference", inferenceTab.panel()),
    stylesheets=["""
    .bk-tab { 
        background: #f0f0f0;
        border-radius: 4px 4px 0 0;
        font-size: 14px;
        padding: 8px 16px;
    }
    .bk-tab.bk-active {
        background: white;
        border-top: 2px solid #007bff;
        font-weight: bold;
    }
    .bk-tabs-header {
        background: #e8e8e8;
    }
    .bk-tabs-content {
        border: 1px solid #ccc;
        padding: 10px;
    }
    """]
)
template.main[:] = [
    pn.Column(tabs, sizing_mode="stretch_both")
]
template.servable()

