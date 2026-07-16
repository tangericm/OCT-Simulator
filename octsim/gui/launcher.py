"""octsim GUI — a single-window PySide6 app to configure and generate OCT data.

Mirrors the Line-Field-OCT app's patterns: a scrollable column of ``QGroupBox``
parameter sections (text/combo/checkbox controls), a live derived-parameter readout
from :meth:`SimConfig.resolve`, a background ``threading``+``queue``+``QTimer`` worker
with cancellation, and ``FigurePanel`` tabs for the B-scan, label map, motion trace, and
a pyqtgraph volume/batch browser.
"""

from __future__ import annotations

import queue
import threading
import traceback

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QScrollArea, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from ..config import SimConfig
from ..generate import generate_dataset, simulate
from .widgets import FigurePanel, StackBrowser

# (section title, [(label, attr, kind, options?)])
SECTIONS = [
    ("Source / Axial", [
        ("Wavelength (nm)", "center_wavelength_nm", "float", None),
        ("Bandwidth FWHM (nm) [0=derive]", "bandwidth_fwhm_nm", "float", None),
        ("Axial resolution (µm) [0=derive]", "axial_resolution_um", "float", None),
        ("Pixels / spectrum", "pixels_per_spectrum", "int", None),
        ("Axial range (µm) [0=derive]", "axial_range_um", "float", None),
        ("Axial sampling (µm) [0=derive]", "axial_sampling_um", "float", None),
    ]),
    ("Scan geometry", [
        ("Pattern", "scan_pattern", "combo", ["bscan", "volume"]),
        ("Lateral FOV (µm)", "lateral_fov_um", "float", None),
        ("Lines / frame", "lines_per_frame", "int", None),
        ("Lateral resolution (µm) [0=beam]", "lateral_resolution_um", "float", None),
        ("Beam waist (µm)", "beam_waist_um", "float", None),
        ("Slow-axis FOV (µm)", "slow_fov_um", "float", None),
        ("Frames (volume)", "n_frames", "int", None),
    ]),
    ("Speed / Timing", [
        ("Line / sweep rate (Hz)", "line_rate_hz", "float", None),
        ("Duty cycle", "duty_cycle", "float", None),
        ("Flyback (s)", "flyback_time_s", "float", None),
    ]),
    ("Dispersion / Fall-off", [
        ("Dispersion (0=sharp, ±=stretch)", "dispersion", "float", None),
        ("Fall-off 6 dB depth (µm) [0=off]", "falloff_6db_um", "float", None),
    ]),
    ("Noise / Artifacts", [
        ("Noise-free", "noise_free", "bool", None),
        ("Noise level", "noise_level", "float", None),
        ("Shot", "shot", "float", None),
        ("RIN", "rin", "float", None),
        ("Read", "read", "float", None),
        ("Target SNR (dB) [0=off]", "target_snr_db", "float", None),
        ("Motion level (0=still)", "motion_level", "float", None),
        ("Axial motion (µm)", "axial_motion_um", "float", None),
        ("Phase drift (rad)", "phase_drift_rad", "float", None),
        ("Cardiac", "cardiac", "bool", None),
        ("Microsaccade", "microsaccade", "bool", None),
    ]),
    ("Anatomy / Scene", [
        ("Anatomy", "anatomy", "combo", ["macula", "onh", "periphery", "mirror", "paper", "cornea"]),
        ("Sample depth (µm)", "sample_depth_um", "float", None),
        ("Lateral offset (µm)", "sample_lateral_offset_um", "float", None),
        ("Randomize", "randomize", "bool", None),
    ]),
    ("Batch", [
        ("N samples", "n_samples", "int", None),
        ("Seed", "seed", "int", None),
    ]),
]


class Launcher(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("octsim — OCT simulation platform")
        self.resize(1280, 860)
        self._widgets: dict[str, QWidget] = {}
        self._kinds: dict[str, str] = {}
        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self.sample = None
        self.batch = None
        self._build()
        self._apply_config(SimConfig())
        self._update_derived()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(80)

    # ---------------------------------------------------------------- construction
    def _build(self) -> None:
        outer = QHBoxLayout(self)

        # left: scrollable parameters
        left = QVBoxLayout()
        bar = QHBoxLayout()
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.clicked.connect(self._generate_one)
        self.batch_btn = QPushButton("Generate N")
        self.batch_btn.clicked.connect(self._generate_batch)
        self.rand_btn = QPushButton("Randomize")
        self.rand_btn.clicked.connect(self._randomize)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(lambda: self.cancel.set())
        self.stop_btn.setEnabled(False)
        for b in (self.gen_btn, self.batch_btn, self.rand_btn, self.stop_btn):
            bar.addWidget(b)
        left.addLayout(bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        vbox = QVBoxLayout(holder)
        for title, fields in SECTIONS:
            vbox.addWidget(self._section(title, fields))
        vbox.addStretch(1)
        scroll.setWidget(holder)
        left.addWidget(scroll, 1)

        self.derived = QLabel()
        self.derived.setWordWrap(True)
        self.derived.setStyleSheet("font-family: monospace; font-size: 11px;")
        dgroup = QGroupBox("Derived parameters")
        dl = QVBoxLayout(dgroup)
        dl.addWidget(self.derived)
        left.addWidget(dgroup)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        left.addWidget(self.progress)
        left.addWidget(QLabel("octsim v0.2"))

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(440)
        outer.addWidget(left_widget)

        # right: figure tabs
        self.tabs = QTabWidget()
        self.fig_bscan = FigurePanel()
        self.fig_label = FigurePanel()
        self.fig_motion = FigurePanel()
        self.browser = StackBrowser()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.fig_bscan, "B-scan")
        self.tabs.addTab(self.fig_label, "Label map")
        self.tabs.addTab(self.fig_motion, "Motion")
        self.tabs.addTab(self.browser, "Volume / batch")
        self.tabs.addTab(self.log, "Log")
        outer.addWidget(self.tabs, 1)

    def _section(self, title: str, fields) -> QGroupBox:
        box = QGroupBox(title)
        grid = QGridLayout(box)
        for idx, (label, attr, kind, options) in enumerate(fields):
            row = idx
            if kind == "bool":
                w = QCheckBox(label)
                grid.addWidget(w, row, 0, 1, 2)
            elif kind == "combo":
                w = QComboBox()
                w.addItems(options)
                w.currentIndexChanged.connect(self._update_derived)
                grid.addWidget(QLabel(label), row, 0)
                grid.addWidget(w, row, 1)
            else:
                w = QLineEdit()
                w.editingFinished.connect(self._update_derived)
                grid.addWidget(QLabel(label), row, 0)
                grid.addWidget(w, row, 1)
            self._widgets[attr] = w
            self._kinds[attr] = kind
        return box

    # ---------------------------------------------------------------- config <-> UI
    def _make_config(self) -> SimConfig:
        data = {}
        for attr, w in self._widgets.items():
            kind = self._kinds[attr]
            if kind == "bool":
                data[attr] = w.isChecked()
            elif kind == "combo":
                data[attr] = w.currentText()
            elif kind == "int":
                data[attr] = int(round(float(w.text() or "0")))
            else:
                data[attr] = float(w.text() or "0")
        return SimConfig(**data).normalized()

    def _apply_config(self, cfg: SimConfig) -> None:
        for attr, w in self._widgets.items():
            val = getattr(cfg, attr)
            kind = self._kinds[attr]
            if kind == "bool":
                w.setChecked(bool(val))
            elif kind == "combo":
                w.setCurrentText(str(val))
            else:
                w.setText(f"{val:g}")

    def _update_derived(self) -> None:
        try:
            rc = self._make_config().resolve()
        except Exception as exc:  # invalid combo -> show message
            self.derived.setText(f"⚠ {exc}")
            return
        d = rc.derived
        self.derived.setText(
            f"axial res   {d['axial_resolution_um']:.2f} µm    "
            f"range {d['axial_range_um']:.0f} µm    pitch {d['axial_sampling_um']:.2f} µm\n"
            f"bandwidth   {d['bandwidth_fwhm_nm']:.1f} nm\n"
            f"lateral res {d['lateral_resolution_um']:.2f} µm    "
            f"fast pitch {d['fast_pitch_um']:.2f} µm    slow pitch {d['slow_pitch_um']:.2f} µm\n"
            f"T_int       {d['integration_time_us']:.3f} µs    "
            f"frame rate {d['frame_rate_hz']:.1f} Hz    acq {d['acquisition_time_s']*1e3:.1f} ms")

    # ---------------------------------------------------------------- worker
    def _busy(self, on: bool) -> None:
        for b in (self.gen_btn, self.batch_btn, self.rand_btn):
            b.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        self.progress.setVisible(on)

    def _start(self, work, label: str) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.cancel.clear()
        self._busy(True)
        self._log(label)

        def target():
            try:
                payload = work()
                if self.cancel.is_set():
                    self.queue.put(("canceled", None))
                else:
                    self.queue.put(("done", payload))
            except Exception as exc:
                self.queue.put(("error", f"{exc}\n{traceback.format_exc()}"))

        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def _generate_one(self) -> None:
        try:
            cfg = self._make_config()
        except Exception as exc:
            self._log(f"invalid settings: {exc}")
            return
        self._start(lambda: ("sample", simulate(cfg)), "Generating sample…")

    def _generate_batch(self) -> None:
        try:
            cfg = self._make_config()
        except Exception as exc:
            self._log(f"invalid settings: {exc}")
            return
        n = cfg.n_samples
        self._start(lambda: ("batch", list(generate_dataset(cfg, n))), f"Generating {n} samples…")

    def _randomize(self) -> None:
        w = self._widgets["seed"]
        w.setText(str(int(float(w.text() or "0")) + 1))
        self._generate_one()

    def _poll(self) -> None:
        try:
            tag, payload = self.queue.get_nowait()
        except queue.Empty:
            return
        self._busy(False)
        if tag == "canceled":
            self._log("canceled.")
        elif tag == "error":
            self._log(f"ERROR: {payload}")
        elif tag == "done":
            kind, data = payload
            if kind == "sample":
                self.sample = data
                self.batch = None
                self._show_sample(data)
                self._log(f"done — {data.params}")
            else:
                self.batch = data
                self.sample = data[0] if data else None
                if data:
                    self._show_sample(data[0])
                    stack = np.stack([s.bscan_db[0] if s.is_volume else s.bscan_db for s in data])
                    self.browser.set_stack(stack)
                self._log(f"done — {len(data)} samples")

    # ---------------------------------------------------------------- display
    def _show_sample(self, s) -> None:
        extent = [s.x_axis_um[0], s.x_axis_um[-1], s.depth_axis_um[-1], s.depth_axis_um[0]]
        self.fig_bscan.clear()
        ax = self.fig_bscan.figure.add_subplot(111)
        ax.imshow(s.bscan_db, cmap="gray", aspect="auto", vmin=-55, vmax=0, extent=extent)
        ax.set_title(f"{s.params.get('anatomy', '')}  ({'volume' if s.is_volume else 'B-scan'})")
        ax.set_xlabel("lateral (µm)")
        ax.set_ylabel("optical depth (µm)")
        self.fig_bscan.draw()

        self.fig_label.clear()
        axl = self.fig_label.figure.add_subplot(111)
        axl.imshow(s.labels2d, cmap="nipy_spectral", aspect="auto", extent=extent)
        axl.set_title("layer label map (free ground truth)")
        axl.set_xlabel("lateral (µm)")
        self.fig_label.draw()

        self.fig_motion.clear()
        axm = self.fig_motion.figure.add_subplot(111)
        if s.motion_dz_um is not None:
            axm.plot(s.motion_dz_um, lw=0.8)
            axm.set_title("bulk axial motion Δz(t) across the acquisition")
            axm.set_xlabel("A-scan index"); axm.set_ylabel("Δz (µm)")
        else:
            axm.text(0.5, 0.5, "no motion (motion_level = 0)", ha="center", va="center")
            axm.set_axis_off()
        self.fig_motion.draw()

        if s.is_volume:
            self.browser.set_stack(s.volume_db)

    def _log(self, msg: str) -> None:
        self.log.append(msg)


def main() -> int:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    win = Launcher()
    win.show()
    return app.exec()
