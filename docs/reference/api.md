# API reference

Autodoc is configured and ready, but no modules are wired in yet. This page holds
the template plus the reasoning, so that turning it on is a copy-and-paste job.

## Turning it on

Sphinx cannot document what it cannot import. Two things are already handled for
you in `docs/conf.py`:

- The repository root is on `sys.path`, so top-level modules like
  `earth2StudioRunner` are importable.
- Every runtime dependency — `panel`, `torch`, `xarray`, `earth2studio`, and the
  rest — is listed in `autodoc_mock_imports`, because none of them are installed on
  the Read the Docs builder. Import a module that pulls in an unmocked dependency
  and the build will report `Failed to import`.

To document a module, add a block like the one below to this page and change the
fence language from `rst` to `{eval-rst}` so Sphinx executes it:

```rst
.. automodule:: earth2StudioRunner
   :members:
```

Bring modules in one at a time and confirm the build is clean after each. Enabling
all nine at once against partially documented modules produces a very long page and
a wall of warnings that hides the one real problem.

## Suggested order

Work outward from the modules with the most stable public surface:

| Order | Module | Why |
| --- | --- | --- |
| 1 | `milesCreditRunner` | Defines the abstract `ModelRunner` base — the contract every runner implements, and the thing a contributor most needs documented. |
| 2 | `earth2StudioRunner` | `MODEL_ENV_MAP` and the dispatch logic. |
| 3 | `dataset_metadata` | Convention detection; small, self-contained, high value. |
| 4 | `datasetSelector2` | Browser widget API. |
| 5 | `earth2StudioPlot`, `era5_plot`, `datasetPlot` | Plot renderers. |
| 6 | `inferenceTab`, `app_layout` | Largest and most UI-coupled; least useful as generated docs. |

## Docstring conventions

Napoleon is enabled for both Google and NumPy style. Pick one per module and stay
consistent within it.

```python
def snap_to_cycle(when: datetime) -> datetime:
    """Snap a timestamp to the nearest GDAS analysis cycle.

    GFS/GDAS analyses are published at 00, 06, 12, and 18 UTC only, so an
    arbitrary user-supplied time has to be resolved to one of them before it
    can be used as an initial condition.

    Args:
        when: The requested initialization time.

    Returns:
        The nearest 00, 06, 12, or 18 UTC time.

    Example:
        >>> snap_to_cycle(datetime(2026, 6, 1, 9, 40))
        datetime.datetime(2026, 6, 1, 6, 0)
    """
```

Two house conventions worth matching:

- **Public methods before private methods** in a class body. Generated docs use
  `member-order: bysource`, so source order is what the reader sees.
- **Complete literal values** in examples — no `...` placeholders. An example the
  reader cannot paste and run is not an example.

## Documenting a single class

When a module is large but one class in it matters, skip `automodule` and pull the
class directly:

```rst
.. autoclass:: earth2StudioRunner.Earth2StudioRunner
   :members:
   :show-inheritance:
```

This keeps the page focused and avoids documenting module-level helpers that are
not part of the public surface.
