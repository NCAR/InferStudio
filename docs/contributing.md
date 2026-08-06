# Contributing

## Building the docs locally

Always build locally before pushing. Read the Docs is slow to iterate against and
its failure output is harder to read than Sphinx's.

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r docs/requirements.txt

sphinx-build -b html docs docs/_build/html
python -m http.server -d docs/_build/html 8000
```

Then open <http://localhost:8000>.

For a tighter loop, add `sphinx-autobuild` and rebuild on save:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

Check for broken cross-references before opening a PR:

```bash
sphinx-build -b linkcheck docs docs/_build/linkcheck
sphinx-build -b html -n -W docs docs/_build/html   # nitpicky, warnings as errors
```

## Read the Docs setup

One-time, per project:

1. Sign in to Read the Docs and import `NCAR/InferStudio`.
2. Confirm it detects `.readthedocs.yaml` at the repository root. If the build
   fails immediately with a config error, the file is in the wrong place — it must
   be at the root, not in `docs/`.
3. Enable **Build pull requests for this project** in *Admin → Settings*. PR
   previews catch documentation breakage before it lands on `main`.
4. Under *Admin → Automation Rules*, add a rule to activate and build new tags so
   tagged releases get their own versioned docs.
5. Set the default version to `stable` once you have tagged a release, or leave it
   at `latest` while the project is pre-release.

Once `fail_on_warning: true` is set in `.readthedocs.yaml`, a broken cross-reference
will fail the build rather than silently produce a dead link. Turn it on as soon as
the build is warning-clean.

## Documentation style

- **Markdown (MyST)** for prose. reStructuredText is accepted where a directive
  has no MyST equivalent.
- **Second person**, present tense. "Select the model", not "The user should
  select the model".
- **State the failure mode.** For anything with a non-obvious constraint — the
  00/06/12/18 UTC snapping, the `vis` queue cap, the capital `S` in
  `earth2StudioPlot` — say what goes wrong when it is violated. A rule without its
  consequence gets ignored.
- **Runnable examples.** Complete literal values, no ellipsis placeholders.
- **One idea per admonition.** Do not stack four warnings in a row; readers skip
  blocks of them wholesale.

## Adding a page

1. Create the file in the appropriate section directory.
2. Add it to the `toctree` in `docs/index.md`. A page absent from a toctree builds
   but is unreachable, and Sphinx warns about it.
3. Cross-reference it from at least one related page. Isolated pages do not get
   read.

## Code changes

Conventions the maintainers follow:

- Public methods before private methods in class bodies.
- Never mutate a Panel or Bokeh widget from a background thread — schedule through
  the document's next-tick callback. Direct mutation deadlocks the server.
- Verify with `grep -n` that an edit actually landed on disk before investigating
  why the behaviour did not change. This is especially important for the generated
  inference scripts, where a change made outside the f-string looks correct and has
  no effect.
- Layout fixes that Panel's `sizing_mode` will not deliver belong in
  `static/styles.css`.

## Reporting a documentation bug

Open an issue at <https://github.com/NCAR/InferStudio/issues> with the page URL and
what you expected. Documentation issues that describe a *wrong* instruction are
high priority — a user following a broken instruction burns GPU allocation.
