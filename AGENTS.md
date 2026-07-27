# AGENTS.md

## Cursor Cloud specific instructions

`octsim` is a pure-Python OCT (optical coherence tomography) simulation platform — a
library + CLI + PySide6 desktop GUI. There is **no server/database**; nothing needs to be
kept running in the background. Install is editable (`pip install -e ".[dev,gui,viz]"`,
handled by the startup update script).

Non-obvious notes for developing here:

- **Console scripts are not on PATH.** `pip install --user` puts the `octsim` and `pytest`
  entry points in `~/.local/bin`, which is not on PATH in this environment. Invoke via the
  module form instead: `python3 -m pytest`, `python3 -m octsim.gui.app ...`.
- **The full test suite is slow (~6 min).** `python3 -m pytest -q` runs ~113 physics/anatomy
  tests. For a fast sanity check use the headless self-test: `python3 -m octsim.gui.app --selftest`
  (exits 0/1). Tests are data-free — anything needing real OCT scans or a display skips cleanly.
- **The wavelength physics check is a separate CI job:** `python3 tests/test_wavelength.py`.
- **GUI requires a display and Qt system libs.** Run it with `DISPLAY=:1 python3 -m octsim.gui.app`
  (or `--gui`). The needed Qt/EGL/xcb system libraries are baked into the VM image; if the GUI
  fails to import with `libEGL.so.1: cannot open shared object file`, the Qt runtime libs are
  missing. For headless GUI tests use `QT_QPA_PLATFORM=offscreen python3 tests/test_gui_smoke.py`.
- **Headless data generation** (no Qt import): `python3 -m octsim.gui.app --generate N --anatomy macula --pattern volume --frames K`.
- **Realism-validation scripts** under `scripts/` (e.g. `validate_realism.py`) need real OCT
  corpora pointed to by `$OCT_DATA_ROOT`; raw scans are never committed, so these are optional
  and not runnable without external data.
- There is no configured linter/formatter (no ruff/flake8/black config); "lint" here is just the
  pytest suite + physics assertions.
