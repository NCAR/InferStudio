"""WeatherBench2-style verification for the Statistics tab's credit-verify
sub-tab: latitude-weighted RMSE and bias against GFS analysis, globally and
for the standard WB2 latitude bands, using credit-verify's scoring code
(vendored in third_party/credit_verify).

Fetching, saving and the compute button all come from ForecastStatsPanel;
this only swaps in the scores and the plots. Results are saved separately
from the basic stats, under <Model>/stats/wb2/.

ACC isn't computed: it needs an ERA5 climatology, which InferStudio doesn't
have access to yet.
"""

import numpy as np
import panel as pn
import matplotlib.pyplot as plt

from usage_log import log_event
from third_party.credit_verify.deterministic import deterministic_scores, latitude_slices
from visualization.earth2StudioVars import _resolve_dim, LAT_NAMES, LON_NAMES
from visualization.forecastStatsPanel import (
    ForecastStatsPanel, on_truth_grid, STATS_DIRNAME)

REGION_LABELS = {
    "global": "Global",
    "tropics": "Tropics (20°S-20°N)",
    "n_extratropics": "N. Extratropics (20-90°N)",
    "s_extratropics": "S. Extratropics (90-20°S)",
}
COLUMNS = tuple(f"{metric}_{region}" for metric in ("rmse", "bias")
                for region in latitude_slices)


def wb2_scores(model_da, truth_da):
    """credit-verify's deterministic scores for one lead time:
    {rmse,bias}_{global,tropics,n_extratropics,s_extratropics}."""
    model_da = on_truth_grid(model_da, truth_da)
    # Same values, so arithmetic lines the two up by position as the basic
    # stats do, even where float coordinates differ in the last digit.
    model_da = model_da.assign_coords(
        {d: truth_da[d] for d in truth_da.dims if d in truth_da.coords})

    def prepare(da):
        # credit-verify wants latitude/longitude and a time dimension, and
        # selects its regions with south-to-north slices, which come out
        # empty on GFS's north-to-south latitudes.
        lat, lon = _resolve_dim(da, *LAT_NAMES), _resolve_dim(da, *LON_NAMES)
        da = da.rename({lat: "latitude", lon: "longitude"})
        return da.sortby("latitude").expand_dims("time")

    scores = deterministic_scores(prepare(model_da), prepare(truth_da))
    return {c: scores[c] for c in COLUMNS}


class WB2StatsPanel(ForecastStatsPanel):
    """ForecastStatsPanel scored the WeatherBench2 way. A region selector
    picks which latitude band the plots show; all bands are computed and
    saved together."""

    TITLE = "WeatherBench2 Verification (credit-verify)"
    METHOD = "credit-verify"
    STATS_SUBDIR = f"{STATS_DIRNAME}/wb2"
    STATS_VERSION = 1
    COLUMNS = COLUMNS
    score = staticmethod(wb2_scores)

    def __init__(self, controls, dataset_key, metadata, **params):
        # Before super().__init__, which renders any saved stats.
        self.region_picker = pn.widgets.Select(
            options={label: key for key, label in REGION_LABELS.items()},
            value="global", width=220, margin=(5, 10))
        self.region_picker.param.watch(self._on_region_change, "value")
        super().__init__(controls, dataset_key, metadata, **params)

    def _controls(self):
        return [self.region_picker]

    def _on_region_change(self, event):
        log_event("stats_region", suite=self.dataset_key,
                  variable=self.controls.var_name, region=event.new)
        if self.results:
            self._render(self.results, self.last_var_name)

    def _render(self, results, var_name):
        region = self.region_picker.value
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for model, df in results.items():
            days = df["lead_hours"] / 24
            axes[0].plot(days, df[f"rmse_{region}"], marker="o", label=model)
            axes[1].plot(days, df[f"bias_{region}"], marker="o", label=model)

        axes[0].set_title("RMSE (latitude-weighted)")
        axes[0].set_ylabel(var_name)
        axes[1].set_title("Bias (forecast - analysis)")
        axes[1].axhline(0, color="0.6", linewidth=0.8)
        for ax in axes:
            ax.set_xlabel("Lead Time (days)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"WeatherBench2 scores vs GFS analysis — {var_name}, "
                     f"{REGION_LABELS[region]}")
        fig.tight_layout()

        self.plot_pane.object = fig
        plt.close(fig)
