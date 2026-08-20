"""Sphinx configuration for the InferStudio documentation."""

import os
import sys
from datetime import date

# -- Path setup --------------------------------------------------------------
# Make the application modules importable so autodoc can find them. Adjust the
# relative path if the InferStudio sources move out of the repository root.
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "InferStudio"
author = "NSF NCAR"
copyright = f"{date.today().year}, University Corporation for Atmospheric Research"

# Keep this in sync with the tag you release from; RTD shows the tag name
# separately, so a short marketing version is fine here.
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST -------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- autodoc ----------------------------------------------------------------
# None of the runtime dependencies are installed on the Read the Docs builder,
# so every heavyweight import has to be mocked or autodoc will fail to import
# the modules it documents.
autodoc_mock_imports = [
    "panel",
    "param",
    "bokeh",
    "holoviews",
    "hvplot",
    "geoviews",
    "cartopy",
    "matplotlib",
    "numpy",
    "pandas",
    "xarray",
    "netCDF4",
    "zarr",
    "dask",
    "scipy",
    "torch",
    "earth2studio",
    "credit",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "panel": ("https://panel.holoviz.org/", None),
}

# -- HTML output ------------------------------------------------------------
html_theme = "furo"
html_title = "InferStudio"
html_static_path = ["_static", "../static/logo"]
html_last_updated_fmt = "%Y-%m-%d"

html_theme_options = {
    "light_logo": "wordmark_light.png",
    "dark_logo": "wordmark_dark.png",
    "source_repository": "https://github.com/NCAR/InferStudio/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/NCAR/InferStudio",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
                ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 "
                "1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82."
                "63-.18 1.31-.27 1.98-.27.67 0 1.35.09 1.98.27 1.53-1.04 2.2-.82 "
                "2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 "
                '3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.'
                '15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

html_favicon = "../static/logo/favicon.ico"
# html_css_files = ["custom.css"]
# html_logo = "_static/inferstudio-logo.svg"
# html_favicon = "_static/favicon.ico"
