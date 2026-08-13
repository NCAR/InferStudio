#!/usr/bin/env python
# coding: utf-8

# InferStudio — launched automatically by NCAR's Open OnDemand "InferStudio"
# app (see template/script.sh.erb in the OOD app bundle for the actual
# `panel serve` invocation, including the URL --prefix OOD requires).
#
# To run manually for local testing (outside OOD), from this directory:
#     panel serve panel_app.py --show

from pathlib import Path
import panel as pn

from app_layout import build_app

DATA_DIR = Path("/glade/derecho/scratch/pearse/CREDIT/RAW_OUTPUT/panelTest/")

pn.extension(
    'modal',
    raw_css=[
        ".bk-btn-group { flex-wrap: wrap !important; max-width: 600px; }",
        ".bk-btn-group button { border-radius: 4px !important; margin: 2px; }",
    ],
    notifications=True,
)

template = build_app(DATA_DIR)
template.servable()
