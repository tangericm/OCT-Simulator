"""GUI smoke test — builds the window offscreen and drives one generation.

Skips cleanly if PySide6 / the Qt platform plugin is unavailable. Run with::

    QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_gui_builds_and_generates():
    try:
        from PySide6.QtWidgets import QApplication
        from octsim.gui.launcher import Launcher
    except Exception as exc:  # pragma: no cover
        print(f"SKIP gui smoke ({exc})")
        return

    app = QApplication.instance() or QApplication([])
    win = Launcher()
    # small, fast config
    win._widgets["pixels_per_spectrum"].setText("512")
    win._widgets["lines_per_frame"].setText("48")
    win._widgets["lateral_fov_um"].setText("900")
    win._update_derived()
    assert "axial res" in win.derived.text()

    win._generate_one()
    deadline = time.time() + 30
    while win.sample is None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert win.sample is not None, "generation did not complete"
    assert win.sample.bscan.shape[0] == 256
    # the figure tabs rendered without error
    assert win.fig_bscan.figure.axes
    print("ok  test_gui_builds_and_generates")


if __name__ == "__main__":
    test_gui_builds_and_generates()
    print("\nGUI smoke passed")
