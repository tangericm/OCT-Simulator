"""Reusable GUI widgets: a matplotlib panel and an optional pyqtgraph stack browser."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as _pg
except Exception:  # pragma: no cover - optional
    _pg = None


class FigurePanel(QWidget):
    """A matplotlib ``Figure`` embedded in Qt with a navigation toolbar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self) -> None:
        self.figure.clear()

    def draw(self) -> None:
        self.canvas.draw_idle()


class StackBrowser(QWidget):
    """A pyqtgraph ``ImageView`` stack scrubber (frames of a volume or a batch).

    Falls back to a plain label when pyqtgraph is unavailable, so the GUI still runs.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if _pg is not None:
            self.view = _pg.ImageView()
            layout.addWidget(self.view)
        else:  # pragma: no cover
            self.view = None
            layout.addWidget(QLabel("pyqtgraph not installed — stack browser unavailable."))

    def set_stack(self, stack) -> None:
        """``stack`` is ``[n_frames, n_z, n_x]``; shown transposed so depth is vertical."""
        if self.view is None or stack is None:
            return
        # pyqtgraph shows [t, x, y]; give it [frame, x, depth] so depth is the vertical axis
        self.view.setImage(stack.transpose(0, 2, 1), autoLevels=True, autoRange=True)
