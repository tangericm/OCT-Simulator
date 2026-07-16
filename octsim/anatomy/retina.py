"""Physiologically-layered retina, modelled in 3D (enface-parameterized).

The retina is a stack of layers with **reasonable physiological thicknesses** (total
neurosensory retina ≈ 270 µm at the parafovea) and the characteristic OCT
hyper/hypo-reflective pattern. The model is 3D: :meth:`RetinaModel.cross_section`
renders the layer boundaries at a given slow-axis position ``y``, so a volume sweeps a
consistent fovea / optic-nerve-head / vessel field.

Geometry:

* **Foveal pit** — the inner layers (NFL…OPL) thin toward the foveal centre while the
  photoreceptor/RPE complex stays anchored at a fixed depth, so the inner surface
  (ILM) dips into the pit and the outer retina stays flat — anatomically correct.
  The ONL thickens toward the centre (the foveal photoreceptor bulge).
* **Optic nerve head** — inside the disc the neuroretinal layers taper out and the
  surface is excavated (the cup).
* **Blood vessels** — enface paths at a superficial plexus depth, each casting a
  shadow (:class:`octsim.features.BloodVessel`).

Optical properties (refractive index, backscatter, scattering) are **illustrative and
calibratable** — there is no validated per-layer retinal optical-property table; these
are seeded to reproduce the qualitative OCT layer pattern and should be tuned against
real macular scans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features import BloodVessel

VITREOUS_N = 1.336


@dataclass(frozen=True)
class RetinaLayer:
    """One retinal layer. ``group`` drives the foveal/ONH geometry response."""

    name: str
    thickness_um: float          # parafoveal physical thickness
    n: float
    backscatter: float
    mu_s_per_mm: float
    group: str                   # nfl | inner | onl | photoreceptor | rpe | choroid
    interface_reflectivity: float | None = None


# Vitreal -> scleral. Thicknesses ~ clinical parafoveal norms (µm); the bright/dark
# alternation (NFL/IPL/OPL/EZ/RPE bright; GCL/INL/ONL dark) is the meaningful part.
RETINA_LAYERS: list[RetinaLayer] = [
    RetinaLayer("NFL", 30.0, 1.38, 4.5e-2, 4.0, "nfl", interface_reflectivity=0.045),
    RetinaLayer("GCL", 35.0, 1.38, 1.0e-2, 2.0, "inner"),
    RetinaLayer("IPL", 40.0, 1.38, 3.0e-2, 3.0, "inner"),
    RetinaLayer("INL", 30.0, 1.38, 9.0e-3, 2.0, "inner"),
    RetinaLayer("OPL", 25.0, 1.38, 3.2e-2, 3.0, "inner"),
    RetinaLayer("ONL", 75.0, 1.38, 7.0e-3, 1.8, "onl"),
    RetinaLayer("ELM", 5.0, 1.38, 3.0e-2, 3.0, "photoreceptor"),
    RetinaLayer("EZ", 8.0, 1.40, 8.0e-2, 5.0, "photoreceptor"),
    RetinaLayer("IZ", 8.0, 1.40, 5.0e-2, 4.0, "photoreceptor"),
    RetinaLayer("RPE", 15.0, 1.40, 9.0e-2, 6.0, "rpe"),
    RetinaLayer("choroid", 250.0, 1.38, 2.5e-2, 3.0, "choroid"),
]
#: total neurosensory retina (NFL..RPE), µm — the flat posterior anchor uses this.
RETINA_TOTAL_UM = sum(ly.thickness_um for ly in RETINA_LAYERS if ly.group != "choroid")


@dataclass
class CrossSection:
    """Everything :func:`octsim.sample.build_reflectivity_from_boundaries` needs."""

    opt_b: np.ndarray            # [L+1, n_x] optical boundary depths
    phys_b: np.ndarray           # [L+1, n_x] physical boundary depths
    layer_n: np.ndarray
    layer_backscatter: np.ndarray
    layer_mu_t: np.ndarray
    layer_interface: list
    top_n: float
    features: list
    label_names: list


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class RetinaModel:
    """A 3D retinal scene sampled by :meth:`cross_section`."""

    location: str = "macula"
    layers: list = field(default_factory=lambda: list(RETINA_LAYERS))
    top_n: float = VITREOUS_N
    surface_opt_um: float = 200.0        # optical depth of the parafoveal ILM
    center_x_um: float = 500.0           # enface x of the fovea / ONH centre
    center_y_um: float = 0.0             # enface y of the centre
    # fovea
    pit_radius_um: float = 350.0
    onl_bulge: float = 0.6
    # optic nerve head
    onh_radius_um: float = 850.0
    onh_cup_depth_um: float = 320.0
    vessels: list = field(default_factory=list)
    curvature_radius_um: float = 12000.0  # gentle global contour

    def cross_section(self, y_um: float, x_axis_um: np.ndarray) -> CrossSection:
        if self.location == "onh":
            return self._onh_cross_section(y_um, x_axis_um)
        return self._layered_cross_section(y_um, x_axis_um)

    def _contour(self, x: np.ndarray) -> np.ndarray:
        xc = x - x.mean()
        return (xc * xc) / (2.0 * self.curvature_radius_um)

    def _assemble(self, x, t_phys, n_arr, depression, base_opt, backscatter, mu_t,
                  interface, names) -> CrossSection:
        """Stack per-column boundaries from thicknesses, anchored posteriorly."""
        n_x = x.size
        L = t_phys.shape[0]
        opt_thick = n_arr[:, None] * t_phys
        # ILM anchored so the RPE/choroid boundary is flat (macula/periphery); the
        # anchor uses the neurosensory retina (everything above `choroid`).
        ilm_opt = base_opt - opt_thick[:-1].sum(axis=0)
        opt_b = np.zeros((L + 1, n_x))
        opt_b[0] = ilm_opt + depression
        for i in range(L):
            opt_b[i + 1] = opt_b[i] + opt_thick[i]
        phys_b = np.zeros((L + 1, n_x))
        for i in range(L):
            phys_b[i + 1] = phys_b[i] + t_phys[i]
        return CrossSection(opt_b=opt_b, phys_b=phys_b, layer_n=n_arr,
                            layer_backscatter=backscatter, layer_mu_t=mu_t,
                            layer_interface=interface, top_n=self.top_n,
                            features=list(self.vessels), label_names=names)

    def _layered_cross_section(self, y_um: float, x_axis_um: np.ndarray) -> CrossSection:
        x = np.asarray(x_axis_um, float)
        n_x = x.size
        layers = self.layers
        base_t = np.array([ly.thickness_um for ly in layers])[:, None] * np.ones((1, n_x))
        t_phys = base_t.copy()

        if self.location == "macula":
            r_fov = np.sqrt((x - self.center_x_um) ** 2 + (y_um - self.center_y_um) ** 2)
            ss = _smoothstep(0.0, self.pit_radius_um, r_fov)          # 0 centre -> 1 outside
            for i, ly in enumerate(layers):
                if ly.group == "nfl":
                    t_phys[i] = base_t[i] * (0.05 + 0.95 * ss)
                elif ly.group == "inner":
                    t_phys[i] = base_t[i] * ss
                elif ly.group == "onl":
                    t_phys[i] = base_t[i] * (1.0 + self.onl_bulge * (1.0 - ss))

        n_arr = np.array([ly.n for ly in layers])
        base_opt = self.surface_opt_um + float(np.sum(
            [ly.n * ly.thickness_um for ly in layers if ly.group != "choroid"]))
        return self._assemble(
            x, t_phys, n_arr, self._contour(x), base_opt,
            np.array([ly.backscatter for ly in layers]),
            np.array([ly.mu_s_per_mm * 1e-3 for ly in layers]),
            [ly.interface_reflectivity for ly in layers],
            [ly.name for ly in layers])

    def _onh_cross_section(self, y_um: float, x_axis_um: np.ndarray) -> CrossSection:
        """Optic nerve head: outer bands TERMINATE at Bruch's-membrane opening, a
        central cup, RNFL thickening toward the disc, and a medium-reflective
        neural-canal fill (informed by clinical ONH OCT morphology)."""
        x = np.asarray(x_axis_um, float)
        n_x = x.size
        r_onh = np.sqrt((x - self.center_x_um) ** 2 + (y_um - self.center_y_um) ** 2)
        bmo = self.onh_radius_um

        # term_outer: 1 outside the disc, 0 inside — a SHARP termination at the BMO.
        term_outer = _smoothstep(bmo - 40.0, bmo + 40.0, r_onh)
        rim_ramp = 1.0 - _smoothstep(bmo, bmo + 280.0, r_onh)         # RNFL thickens toward disc
        cup = self.onh_cup_depth_um * (1.0 - _smoothstep(0.15 * bmo, bmo, r_onh))

        # layer stack: insert a neural-canal fill just beneath the NFL/rim.
        src = {ly.name: ly for ly in RETINA_LAYERS}
        order = ["NFL", "canal", "GCL", "IPL", "INL", "OPL", "ONL", "ELM", "EZ", "IZ", "RPE", "choroid"]
        base = {"NFL": src["NFL"].thickness_um, "canal": 0.0,
                **{k: src[k].thickness_um for k in order if k not in ("NFL", "canal")}}
        n_map = {**{ly.name: ly.n for ly in RETINA_LAYERS}, "canal": 1.38}
        bs_map = {**{ly.name: ly.backscatter for ly in RETINA_LAYERS}, "canal": 2.2e-2}
        mu_map = {**{ly.name: ly.mu_s_per_mm for ly in RETINA_LAYERS}, "canal": 3.0}
        if_map = {**{ly.name: ly.interface_reflectivity for ly in RETINA_LAYERS}, "canal": None}

        t_phys = np.zeros((len(order), n_x))
        for i, name in enumerate(order):
            if name == "NFL":
                t_phys[i] = base["NFL"] * (1.0 + 2.2 * rim_ramp)      # rim thickening
            elif name == "canal":
                t_phys[i] = 260.0 * (1.0 - term_outer)               # fills the disc
            else:
                t_phys[i] = base[name] * term_outer                  # terminate at BMO

        n_arr = np.array([n_map[name] for name in order])
        # top-down anchor from a cupped ILM (the disc's defining feature is the cup)
        opt_thick = n_arr[:, None] * t_phys
        opt_b = np.zeros((len(order) + 1, n_x))
        opt_b[0] = self.surface_opt_um + cup + self._contour(x)
        for i in range(len(order)):
            opt_b[i + 1] = opt_b[i] + opt_thick[i]
        phys_b = np.zeros((len(order) + 1, n_x))
        for i in range(len(order)):
            phys_b[i + 1] = phys_b[i] + t_phys[i]

        return CrossSection(
            opt_b=opt_b, phys_b=phys_b, layer_n=n_arr,
            layer_backscatter=np.array([bs_map[name] for name in order]),
            layer_mu_t=np.array([mu_map[name] * 1e-3 for name in order]),
            layer_interface=[if_map[name] for name in order],
            top_n=self.top_n, features=list(self.vessels), label_names=list(order))


# --------------------------------------------------------------------------- builders
def default_retina(*, location: str = "macula", surface_opt_um: float = 200.0,
                   center_x_um: float = 500.0, lateral_fov_um: float = 1000.0) -> RetinaModel:
    """A fixed, reasonable retina (no randomness) for previews/tests."""
    vessels = [
        BloodVessel(x0_um=center_x_um - 300.0, y0_um=0.0,
                    depth_opt_um=surface_opt_um + 25.0, caliber_um=70.0, shadow_od=1.1),
        BloodVessel(x0_um=center_x_um + 320.0, y0_um=0.0,
                    depth_opt_um=surface_opt_um + 30.0, caliber_um=55.0, shadow_od=0.9),
    ]
    return RetinaModel(location=location, surface_opt_um=surface_opt_um,
                       center_x_um=center_x_um, vessels=vessels)


def random_retina(rng: np.random.Generator, *, location: str = "macula",
                  surface_opt_um: float = 200.0, lateral_fov_um: float = 1000.0,
                  lateral_offset_um: float = 0.0) -> RetinaModel:
    """Draw a random-but-physiological retina instance for stochastic generation.

    Randomizes per-layer thickness (±15%) and backscatter (±20%), the foveal-pit
    radius / ONL bulge, the ONH cup, the centre position, and a set of vessels.
    """
    layers = []
    for ly in RETINA_LAYERS:
        t = ly.thickness_um * float(rng.uniform(0.85, 1.15))
        b = ly.backscatter * float(rng.uniform(0.8, 1.2))
        layers.append(RetinaLayer(ly.name, t, ly.n, b, ly.mu_s_per_mm, ly.group,
                                  ly.interface_reflectivity))
    center_x = lateral_fov_um / 2.0 + float(lateral_offset_um) + float(rng.uniform(-80, 80))

    n_vessels = int(rng.integers(2, 6))
    plexus = surface_opt_um + float(rng.uniform(15.0, 45.0))
    vessels = []
    for _ in range(n_vessels):
        vx = float(rng.uniform(0.1, 0.9)) * lateral_fov_um
        if location == "macula" and abs(vx - center_x) < 250.0:
            vx += 300.0 * np.sign(vx - center_x + 1e-9)   # keep the fovea avascular
        vessels.append(BloodVessel(
            x0_um=vx, y0_um=float(rng.uniform(-200, 200)),
            depth_opt_um=plexus + float(rng.uniform(-8, 8)),
            caliber_um=float(rng.uniform(30.0, 120.0)),
            slope=float(rng.uniform(-0.4, 0.4)),
            shadow_od=float(rng.uniform(0.7, 1.5))))

    return RetinaModel(
        location=location, layers=layers, surface_opt_um=surface_opt_um, center_x_um=center_x,
        center_y_um=float(rng.uniform(-100, 100)),
        pit_radius_um=float(rng.uniform(280.0, 450.0)),
        onl_bulge=float(rng.uniform(0.4, 0.8)),
        onh_radius_um=float(rng.uniform(700.0, 1000.0)),
        onh_cup_depth_um=float(rng.uniform(250.0, 400.0)),
        vessels=vessels)
