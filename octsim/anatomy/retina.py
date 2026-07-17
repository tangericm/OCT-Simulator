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

from dataclasses import dataclass, field, replace

import numpy as np

from ..features import BloodVessel, ChoroidTexture
from ..tissue import REF_WAVELENGTH_NM, scale_layer

VITREOUS_N = 1.336


@dataclass(frozen=True)
class RetinaLayer:
    """One retinal layer. ``group`` drives the foveal/ONH geometry response.

    Optical properties are stored at the 850 nm reference (:mod:`octsim.tissue`) and
    scaled to the source wavelength at render time; ``water_fraction`` and ``melanin``
    set the wavelength-dependent absorption.
    """

    name: str
    thickness_um: float          # parafoveal physical thickness
    n: float
    backscatter: float           # backscatter amplitude at 850 nm
    mu_s_per_mm: float           # scattering coefficient at 850 nm
    group: str                   # nfl | inner | onl | photoreceptor | rpe | choroid
    interface_reflectivity: float | None = None
    water_fraction: float = 0.65
    melanin: float = 0.0         # pigment (RPE/choroid) — strong NIR absorber


# Foveal avascular zone radius: the central capillary-free region that superficial vessels
# ring but do not cross. OCTA-500 normal 3mm eyes measure a median FAZ area ~0.33 mm²
# (equivalent radius ~324 µm; octsim/data/octa500_macula.json).
FAZ_RADIUS_UM = 320.0

# Vitreal -> scleral. Thicknesses ~ clinical parafoveal norms (µm); the bright/dark
# alternation (NFL/IPL/OPL/EZ/RPE bright; GCL/INL/ONL dark) is the meaningful part.
# Contrast widened and the outer complex brightened to match clinical B-scans.
RETINA_LAYERS: list[RetinaLayer] = [
    RetinaLayer("NFL", 30.0, 1.38, 5.5e-2, 4.5, "nfl", interface_reflectivity=0.02),
    RetinaLayer("GCL", 38.0, 1.38, 6.0e-3, 1.6, "inner"),
    RetinaLayer("IPL", 40.0, 1.38, 2.8e-2, 3.0, "inner"),
    RetinaLayer("INL", 30.0, 1.38, 5.0e-3, 1.5, "inner"),
    RetinaLayer("OPL", 25.0, 1.38, 3.0e-2, 3.0, "inner"),
    RetinaLayer("ONL", 78.0, 1.38, 3.5e-3, 1.2, "onl"),
    RetinaLayer("ELM", 5.0, 1.38, 2.5e-2, 3.0, "photoreceptor"),
    RetinaLayer("EZ", 9.0, 1.41, 1.1e-1, 6.0, "photoreceptor"),
    RetinaLayer("IZ", 7.0, 1.41, 6.0e-2, 5.0, "photoreceptor"),
    RetinaLayer("RPE", 18.0, 1.42, 1.3e-1, 8.0, "rpe", water_fraction=0.5, melanin=1.0),
    RetinaLayer("choroid", 260.0, 1.38, 2.2e-2, 3.0, "choroid", water_fraction=0.6, melanin=0.5),
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
    # realism knobs consumed by the reflectivity builder (0 = off / MATLAB-plain)
    texture_contrast: float = 0.0        # slow intra-layer mottling depth
    interface_jitter: float = 0.0        # lateral break-up of specular interfaces
    vitreous_backscatter: float = 0.0    # faint fill above the ILM (not pure black)
    layer_directional: object = None     # [L, n_x] per-column directional gain (or None)
    boundary_sharpness_um: float = 0.0   # axial transition width (softens the bands)
    texture_field: object = None         # CoherentField for volume-coherent mottling (or None)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Perlin smootherstep (zero 1st and 2nd derivatives at the edges) — gentler."""
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@dataclass
class BoundaryUndulation:
    """Low-frequency spatially-correlated waviness added to every layer boundary.

    Real layer boundaries are gently undulating (few-µm RMS, sub-mm wavelength), not
    exact vertical offsets of one curve. Each boundary gets a SHARED low-frequency
    component (so neighbours co-undulate) plus its OWN independent component (so gaps
    still vary) — the statistical-shape behaviour that removes the mechanically-rigid,
    perfectly-parallel look.
    """

    amp_um: float
    shared: np.ndarray           # [K, 4]  (kx, ky, phase, weight)
    per: np.ndarray              # [n_boundaries, K, 4]

    @classmethod
    def make(cls, n_boundaries: int, rng: np.random.Generator, *, amp_um: float = 4.0,
             k: int = 4, corr_um: float = 450.0) -> "BoundaryUndulation":
        def comps(n):
            f = (2.0 * np.pi) / (corr_um * rng.uniform(0.7, 2.5, (n, k)))
            ang = rng.uniform(0.0, 2.0 * np.pi, (n, k))
            return np.stack([f * np.cos(ang), f * np.sin(ang),
                             rng.uniform(0.0, 2.0 * np.pi, (n, k)),
                             rng.uniform(0.4, 1.0, (n, k))], axis=-1)
        return cls(amp_um, comps(1)[0], comps(n_boundaries))

    @staticmethod
    def _eval(c: np.ndarray, x: np.ndarray, y: float) -> np.ndarray:
        kx, ky, ph, w = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        phase = kx[None, :] * x[:, None] + ky[None, :] * y + ph[None, :]
        return (w[None, :] * np.sin(phase)).sum(axis=1) / np.sqrt((w ** 2).sum())

    def displace(self, i: int, x: np.ndarray, y: float) -> np.ndarray:
        j = min(i, self.per.shape[0] - 1)
        return self.amp_um * (0.6 * self._eval(self.shared, x, y)
                              + 0.7 * self._eval(self.per[j], x, y))


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
    pit_radius_um: float = 700.0
    onl_bulge: float = 0.55
    # optic nerve head
    onh_radius_um: float = 850.0          # Bruch's-membrane-opening half-width (~0.85 mm)
    onh_cup_depth_um: float = 320.0
    disc_dx_um: float = 4760.0            # signed fovea->disc lateral offset (temporal fovea)
    disc_present: bool = True             # include the disc so it appears at a wide FOV
    disc_fill_backscatter: float = 2.2e-2  # neural-canal / prelaminar tissue reflectivity
    vessels: list = field(default_factory=list)
    # global curvature: concave-up posterior-pole bowl (center deepest), saturating so
    # a wide scan stays in-window. `curvature_radius_um` sets the near-fovea radius.
    curvature_radius_um: float = 37500.0
    curvature_sag_um: float = 520.0       # max peripheral rise of the bowl
    curvature_concave: bool = True        # True = bowl (real posterior pole); False = legacy dome
    boundary_sharpness_um: float = 5.0    # axial reflectivity-transition width (softens bands)
    choroid_contrast: float = 0.55        # choroidal vascular lucency depth
    tilt_um_per_mm: float = 0.0           # global B-scan tilt (per-eye)
    pit_asymmetry: float = 0.0            # -1..1 skew of the two foveal walls
    undulation: object = None            # BoundaryUndulation | None
    # data-driven geometry (from a LayerShapeModel): per-layer thickness (µm) on an
    # eccentricity grid; when set it REPLACES the parametric foveal pit with real morphology.
    data_thickness: object = None         # [n_layer, n_ecc] µm, aligned to self.layers
    data_ecc_um: object = None            # [n_ecc] signed eccentricity from center_x_um
    # realism texture (propagated onto every CrossSection; see the builder)
    texture_contrast: float = 0.22       # slow intra-layer mottling depth
    interface_jitter: float = 0.30       # lateral break-up of specular interfaces
    vitreous_backscatter: float = 5.0e-4  # faint vitreous fill (not pure black)
    # directional reflectance
    directional_strength: float = 0.6    # 0 = isotropic; scales SC + Henle effects
    henle_sign: float = 0.0              # -1..1 pupil-offset sign of the HFL asymmetry
    # volume-coherent texture (fixed 3-D fields; sampled per frame -> smooth across a volume)
    mottle_field: object = None          # CoherentField for intra-layer mottling
    choroid_field: object = None         # CoherentField for the choroidal vascular pattern

    def _landmarks(self):
        """Return ``(fovea_x, disc_x_or_None)`` for this scene in absolute enface x (µm)."""
        if self.location == "onh":
            return self.center_x_um - self.disc_dx_um, self.center_x_um
        if self.location == "periphery":
            return self.center_x_um - 6000.0, None      # fovea far off; no disc in view
        disc_x = self.center_x_um + self.disc_dx_um if self.disc_present else None
        return self.center_x_um, disc_x

    def cross_section(self, y_um: float, x_axis_um: np.ndarray,
                      wavelength_nm: float = REF_WAVELENGTH_NM) -> CrossSection:
        x = np.asarray(x_axis_um, float)
        fovea_x, disc_x = self._landmarks()
        # the disc is rendered whenever the scan reaches it (wide FOV) or the scene is
        # ONH-centred; otherwise the fast 11-layer macula path is used unchanged.
        disc_in_view = disc_x is not None and x.size and \
            (x.min() - 1200.0 <= disc_x <= x.max() + 1200.0)
        if disc_in_view or self.location == "onh":
            cross = self._field_cross_section(y_um, x, wavelength_nm, fovea_x, disc_x)
            groups = self._onh_groups(len(cross.label_names))
        else:
            cross = self._layered_cross_section(y_um, x, wavelength_nm)
            groups = [ly.group for ly in self.layers]
        cross.texture_contrast = self.texture_contrast
        cross.interface_jitter = self.interface_jitter
        cross.vitreous_backscatter = self.vitreous_backscatter
        cross.boundary_sharpness_um = self.boundary_sharpness_um
        cross.texture_field = self.mottle_field
        cross.layer_directional = self._directional_gain(cross.opt_b, x, groups)
        return cross

    @staticmethod
    def _onh_groups(n: int) -> list:
        by_name = {ly.name: ly.group for ly in RETINA_LAYERS}
        order = ["NFL", "canal", "GCL", "IPL", "INL", "OPL", "ONL",
                 "ELM", "EZ", "IZ", "RPE", "choroid"]
        return [by_name.get(name, "canal") for name in order[:n]]

    def _directional_gain(self, opt_b: np.ndarray, x: np.ndarray, groups: list):
        """Per-column, per-layer directional reflectance gain ``[L, n_x]`` (or None).

        Physically the backscatter of oriented structures depends on the beam's local
        angle of incidence. Two clinically dominant cues are modelled; everything else
        (RPE especially) stays angle-stable so the EZ/IZ *wings* fade while the RPE
        holds bright:

        * **Photoreceptors (ELM/EZ/IZ)** — Stiles–Crawford: brightest where the band is
          perpendicular to the beam (flat, near the fovea), falling to a floor where the
          layer tilts (periphery, pit walls). Gain ``floor + (1-floor)·exp(-(θ/θ0)²)``.
        * **Henle fibre layer (ONL)** — a pupil-offset-dependent asymmetry about the
          fovea: one side brightens, the other darkens (``1 + sign·a·tanh((x-xc)/w)``).
        """
        if self.directional_strength <= 0:
            return None
        L = len(groups)
        n_x = x.size
        if n_x < 2:
            return None
        gain = np.ones((L, n_x))
        s = float(self.directional_strength)
        dx = (x[-1] - x[0]) / (n_x - 1)
        xc = x - self.center_x_um
        for i, g in enumerate(groups):
            if g == "photoreceptor":
                mid = 0.5 * (opt_b[i] + opt_b[i + 1])
                theta = np.gradient(mid, dx) / self.top_n     # ~tan(incidence), rad
                floor = 1.0 - 0.7 * s
                gain[i] = floor + (1.0 - floor) * np.exp(-(theta / 0.11) ** 2)
            elif g == "onl":
                gain[i] = 1.0 + self.henle_sign * 0.35 * s * np.tanh(xc / 350.0)
        return gain

    @staticmethod
    def _layer_optics(layers, wavelength_nm):
        """Wavelength-scaled ``(backscatter[], mu_t_per_um[])`` for a layer list."""
        bs, mut = [], []
        for ly in layers:
            m, b = scale_layer(ly.mu_s_per_mm, ly.backscatter, wavelength_nm,
                               water_fraction=ly.water_fraction, melanin=ly.melanin)
            bs.append(b)
            mut.append(m)
        return np.array(bs), np.array(mut)

    def _contour(self, x: np.ndarray) -> np.ndarray:
        """Global posterior-pole curvature (optical-depth offset added to every boundary).

        The real posterior pole images **concave-up** (a bowl: the macula sits deepest and
        the retina rises toward the periphery), the opposite of a convex dome. A saturating
        form gives the configured near-fovea radius at the centre but flattens far out, so a
        wide scan curves realistically yet stays inside the depth window. Centred on the
        anatomic centre; a per-eye tilt is added.
        """
        xc = x - self.center_x_um
        R = self.curvature_radius_um
        sag_max = self.curvature_sag_um
        if R <= 0 or sag_max <= 0:
            bowl = np.zeros_like(x, dtype=float)
        else:
            bowl = sag_max * (1.0 - np.exp(-(xc * xc) / (2.0 * R * sag_max)))
        # concave-up: centre deepest (largest offset), periphery shallow (0)
        depth = (bowl.max() - bowl) if self.curvature_concave else bowl
        tilt = self.tilt_um_per_mm * 1e-3 * (x - x.mean())
        return depth + tilt

    def _undulate(self, opt_b: np.ndarray, x: np.ndarray, y_um: float) -> None:
        """Add per-boundary undulation in place, then keep boundaries monotone."""
        if self.undulation is None:
            return
        for i in range(opt_b.shape[0]):
            opt_b[i] = opt_b[i] + self.undulation.displace(i, x, y_um)
        for i in range(1, opt_b.shape[0]):
            opt_b[i] = np.maximum(opt_b[i], opt_b[i - 1] + 0.5)  # min gap; no crossing

    def _assemble(self, x, t_phys, n_arr, depression, base_opt, backscatter, mu_t,
                  interface, names, y_um: float = 0.0) -> CrossSection:
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
        self._undulate(opt_b, x, y_um)
        phys_b = np.zeros((L + 1, n_x))
        for i in range(L):
            phys_b[i + 1] = phys_b[i] + t_phys[i]
        return CrossSection(opt_b=opt_b, phys_b=phys_b, layer_n=n_arr,
                            layer_backscatter=backscatter, layer_mu_t=mu_t,
                            layer_interface=interface, top_n=self.top_n,
                            features=list(self.vessels), label_names=names)

    def _layered_cross_section(self, y_um, x_axis_um, wavelength_nm) -> CrossSection:
        x = np.asarray(x_axis_um, float)
        n_x = x.size
        layers = self.layers
        base_t = np.array([ly.thickness_um for ly in layers])[:, None] * np.ones((1, n_x))
        t_phys = base_t.copy()

        if self.data_thickness is not None:
            # data-driven: interpolate each layer's real thickness profile onto this scan at
            # its RADIAL eccentricity (fast + slow), so the pit recedes across a volume's
            # frames. The pit/bulge come from the shape model, not a formula.
            dt = np.asarray(self.data_thickness, float)
            de = np.asarray(self.data_ecc_um, float)
            for i in range(min(len(layers), dt.shape[0])):
                t_phys[i] = self._radial_interp(dt[i], de, x, self.center_x_um, y_um)
        elif self.location == "macula":
            r_fov = np.sqrt((x - self.center_x_um) ** 2 + (y_um - self.center_y_um) ** 2)
            # per-column pit radius so the two foveal walls can differ (asymmetry);
            # smootherstep gives a gentle, non-peaky parafoveal ridge.
            side = np.tanh((x - self.center_x_um) / 150.0)            # smooth -1..1
            pit_r = self.pit_radius_um * (1.0 + 0.35 * self.pit_asymmetry * side)
            t = np.clip(r_fov / np.maximum(pit_r, 1e-6), 0.0, 1.0)
            ss = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)           # 0 centre -> 1 outside
            for i, ly in enumerate(layers):
                if ly.group == "nfl":
                    t_phys[i] = base_t[i] * (0.02 + 0.98 * ss)
                elif ly.group == "inner":
                    t_phys[i] = base_t[i] * ss
                elif ly.group == "onl":
                    t_phys[i] = base_t[i] * (1.0 + self.onl_bulge * (1.0 - ss))

        n_arr = np.array([ly.n for ly in layers])
        base_opt = self.surface_opt_um + float(np.sum(
            [ly.n * ly.thickness_um for ly in layers if ly.group != "choroid"]))
        bs, mut = self._layer_optics(layers, wavelength_nm)
        cross = self._assemble(
            x, t_phys, n_arr, self._contour(x), base_opt, bs, mut,
            [ly.interface_reflectivity for ly in layers],
            [ly.name for ly in layers], y_um=y_um)
        # choroidal vascular texture (large lucencies below the RPE)
        choroid_idx = [i for i, ly in enumerate(layers) if ly.group == "choroid"]
        if choroid_idx:
            i = choroid_idx[0]
            cross.features.append(ChoroidTexture(
                top_opt=cross.opt_b[i], bot_opt=cross.opt_b[i + 1],
                contrast=self.choroid_contrast, field=self.choroid_field))
        return cross

    _FIELD_ORDER = ["NFL", "canal", "GCL", "IPL", "INL", "OPL", "ONL",
                    "ELM", "EZ", "IZ", "RPE", "choroid"]

    def _radial_interp(self, dt_i, de, x, fovea_x, y_um, blend_um=400.0):
        """Sample a 1-D thickness profile ``dt_i`` (on grid ``de``) at the **radial**
        eccentricity for a frame at slow position ``y_um``.

        Off the foveal B-scan the whole profile is pushed to larger radius
        ``hypot(x-fovea, y-fovea_y)``, so the pit shallows and recedes as a volume's frames
        sweep away from the fovea. The two radial sides are blended over ``blend_um`` around
        the fovea column so an asymmetric profile (e.g. the RNFL nasal/temporal gradient)
        stays continuous — no step at ``x = fovea``."""
        ecc_x = np.asarray(x, float) - fovea_x
        dy = float(y_um) - self.center_y_um
        if abs(dy) < 1e-6:
            return np.interp(ecc_x, de, dt_i)
        r = np.hypot(ecc_x, dy)
        tp = np.interp(r, de, dt_i)          # +radial side
        tm = np.interp(-r, de, dt_i)         # -radial side
        w = 0.5 * (1.0 + np.clip(ecc_x / blend_um, -1.0, 1.0))
        return w * tp + (1.0 - w) * tm

    def _macula_thickness(self, x, y_um):
        """Per-layer macular thickness ``{name: [n_x]}`` for the 11 real layers, data-driven
        (shape model) when available, else the parametric foveal-pit formula."""
        n_x = x.size
        fovea_x, _ = self._landmarks()
        t = {}
        if self.data_thickness is not None:
            de = np.asarray(self.data_ecc_um, float)
            dt = np.asarray(self.data_thickness, float)
            for i, ly in enumerate(self.layers):
                t[ly.name] = (self._radial_interp(dt[i], de, x, fovea_x, y_um)
                              if i < dt.shape[0] else np.full(n_x, ly.thickness_um))
        else:
            r_fov = np.sqrt((x - fovea_x) ** 2 + (y_um - self.center_y_um) ** 2)
            side = np.tanh((x - fovea_x) / 150.0)
            pit_r = self.pit_radius_um * (1.0 + 0.35 * self.pit_asymmetry * side)
            s = np.clip(r_fov / np.maximum(pit_r, 1e-6), 0.0, 1.0)
            ss = s * s * s * (s * (s * 6.0 - 15.0) + 10.0)
            for ly in self.layers:
                base = ly.thickness_um
                if ly.group == "nfl":
                    t[ly.name] = base * (0.02 + 0.98 * ss)
                elif ly.group == "inner":
                    t[ly.name] = base * ss
                elif ly.group == "onl":
                    t[ly.name] = base * (1.0 + self.onl_bulge * (1.0 - ss))
                else:
                    t[ly.name] = np.full(n_x, base)
        return t

    def _field_cross_section(self, y_um, x, wavelength_nm, fovea_x, disc_x) -> CrossSection:
        """One retinal field carrying BOTH the (data-driven) macula and a parametric optic
        disc at their anatomical separation, so a wide scan shows the macula transition into
        the ONH. Top-down anchored from the ILM with a pit-compensation term that keeps the
        RPE/BM flat under the fovea; the disc adds a cup, terminates the layers at Bruch's-
        membrane opening, thickens the RNFL rim, and fills the neural canal.
        """
        n_x = x.size
        order = self._FIELD_ORDER
        by_name = {ly.name: ly for ly in self.layers}
        canal = RetinaLayer("canal", 0.0, 1.38, self.disc_fill_backscatter, 3.0, "canal")
        layers = [canal if nm == "canal" else by_name[nm] for nm in order]

        t_mac = self._macula_thickness(x, y_um)          # 11 real layers, pre-disc
        # parafoveal baseline per neurosensory layer -> the ILM dip (pit) is the thinning
        # relative to it, so the RPE stays flat where the retina is intact.
        neuro = [nm for nm in order if nm not in ("canal", "choroid")]
        pit_depression = np.zeros(n_x)
        for nm in neuro:
            base = float(np.max(t_mac[nm]))
            pit_depression += by_name[nm].n * (base - t_mac[nm])

        cup = np.zeros(n_x)
        t_final = {nm: (t_mac[nm].copy() if nm in t_mac else np.zeros(n_x)) for nm in order}
        t_final["canal"] = np.zeros(n_x)
        if disc_x is not None:
            r_disc = np.sqrt((x - disc_x) ** 2 + (y_um - self.center_y_um) ** 2)
            bmo = self.onh_radius_um
            term = _smoothstep(bmo - 40.0, bmo + 40.0, r_disc)     # 1 outside, 0 inside
            rim = 1.0 - _smoothstep(bmo, bmo + 280.0, r_disc)
            cup = self.onh_cup_depth_um * (1.0 - _smoothstep(0.15 * bmo, bmo, r_disc))
            for nm in neuro:
                t_final[nm] = t_mac[nm] * term                     # terminate at the BMO
            t_final["NFL"] = t_mac["NFL"] * (1.0 + 2.2 * rim) * term  # rim thickening
            t_final["canal"] = 260.0 * (1.0 - term)                # neural-canal fill
            t_final["choroid"] = t_mac.get("choroid", np.full(n_x, by_name["choroid"].thickness_um))

        n_arr = np.array([ly.n for ly in layers])
        t_phys = np.array([t_final[nm] for nm in order])
        opt_b = np.zeros((len(order) + 1, n_x))
        opt_b[0] = self.surface_opt_um + self._contour(x) + pit_depression + cup
        for i in range(len(order)):
            opt_b[i + 1] = opt_b[i] + n_arr[i] * t_phys[i]
        self._undulate(opt_b, x, y_um)
        phys_b = np.zeros((len(order) + 1, n_x))
        for i in range(len(order)):
            phys_b[i + 1] = phys_b[i] + t_phys[i]

        bs, mut = self._layer_optics(layers, wavelength_nm)
        cross = CrossSection(
            opt_b=opt_b, phys_b=phys_b, layer_n=n_arr,
            layer_backscatter=bs, layer_mu_t=mut,
            layer_interface=[ly.interface_reflectivity for ly in layers],
            top_n=self.top_n, features=list(self.vessels), label_names=list(order))
        ci = order.index("choroid")
        cross.features.append(ChoroidTexture(
            top_opt=cross.opt_b[ci], bot_opt=cross.opt_b[ci + 1],
            contrast=self.choroid_contrast, field=self.choroid_field))
        return cross


# --------------------------------------------------------------------------- builders
def default_retina(*, location: str = "macula", surface_opt_um: float = 200.0,
                   center_x_um: float = 500.0, lateral_fov_um: float = 1000.0) -> RetinaModel:
    """A fixed, reasonable retina (no randomness) for previews/tests."""
    vessels = [
        BloodVessel(x0_um=center_x_um - 300.0, y0_um=0.0, caliber_um=70.0, shadow_od=1.1),
        BloodVessel(x0_um=center_x_um + 320.0, y0_um=0.0, caliber_um=55.0,
                    plexus_offset_um=25.0, shadow_od=0.9),
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
        # replace() preserves water_fraction + melanin (positional rebuild dropped them,
        # silently killing RPE/choroid NIR absorption across the whole randomized corpus).
        layers.append(replace(ly,
                              thickness_um=ly.thickness_um * float(rng.uniform(0.85, 1.15)),
                              backscatter=ly.backscatter * float(rng.uniform(0.8, 1.2))))
    center_x = lateral_fov_um / 2.0 + float(lateral_offset_um) + float(rng.uniform(-80, 80))

    n_vessels = int(rng.integers(2, 6))
    vessels = []
    for _ in range(n_vessels):
        vx = float(rng.uniform(0.1, 0.9)) * lateral_fov_um
        if location == "macula" and abs(vx - center_x) < FAZ_RADIUS_UM:
            # ring the fovea rather than cross it: land just outside the avascular zone
            vx = center_x + np.sign(vx - center_x + 1e-9) * (FAZ_RADIUS_UM + float(rng.uniform(10.0, 90.0)))
        cal = float(rng.uniform(30.0, 120.0))
        # large trunks ride the superficial plexus; small vessels can sit deeper
        plexus = 0.0 if cal > 80.0 else float(rng.choice([0.0, 0.0, 25.0, 50.0]))
        vessels.append(BloodVessel(
            x0_um=vx, y0_um=float(rng.uniform(-200, 200)), caliber_um=cal,
            plexus_offset_um=plexus + float(rng.uniform(0.0, 8.0)),
            slope=float(rng.uniform(-0.4, 0.4)),
            shadow_od=float(rng.uniform(0.7, 1.5))))

    undulation = BoundaryUndulation.make(
        len(RETINA_LAYERS) + 2, rng,
        amp_um=float(rng.uniform(2.0, 6.0)), corr_um=float(rng.uniform(350.0, 600.0)))
    texture_contrast = float(rng.uniform(0.16, 0.30))
    choroid_contrast = float(rng.uniform(0.45, 0.62))
    mottle_field, choroid_field = _texture_fields(rng, texture_contrast, choroid_contrast)

    return RetinaModel(
        location=location, layers=layers, surface_opt_um=surface_opt_um, center_x_um=center_x,
        center_y_um=float(rng.uniform(-100, 100)),
        pit_radius_um=float(rng.uniform(550.0, 850.0)),
        onl_bulge=float(rng.uniform(0.4, 0.7)),
        curvature_radius_um=float(rng.uniform(30000.0, 48000.0)),
        curvature_sag_um=float(rng.uniform(420.0, 640.0)),
        boundary_sharpness_um=float(rng.uniform(4.0, 7.0)),
        choroid_contrast=choroid_contrast,
        onh_radius_um=float(rng.uniform(700.0, 1000.0)),
        onh_cup_depth_um=float(rng.uniform(250.0, 400.0)),
        disc_dx_um=float(rng.choice([-1.0, 1.0]) * rng.uniform(4300.0, 5200.0)),
        tilt_um_per_mm=float(rng.uniform(-25.0, 25.0)),
        pit_asymmetry=float(rng.uniform(-0.5, 0.5)),
        undulation=undulation,
        texture_contrast=texture_contrast,
        interface_jitter=float(rng.uniform(0.20, 0.40)),
        vitreous_backscatter=float(rng.uniform(3.0e-4, 7.0e-4)),
        directional_strength=float(rng.uniform(0.45, 0.75)),
        henle_sign=float(rng.choice([-1.0, 0.0, 1.0]) * rng.uniform(0.5, 1.0)),
        mottle_field=mottle_field, choroid_field=choroid_field,
        vessels=vessels)


def _texture_fields(rng, texture_contrast, choroid_contrast):
    """Build the two volume-coherent 3-D texture fields (mottling + choroid) for a scene."""
    from ..sample import CoherentField
    mottle = CoherentField(rng, corr_z_um=18.0, corr_x_um=70.0, corr_y_um=300.0,
                           contrast=texture_contrast)
    choroid = CoherentField(rng, corr_z_um=90.0, corr_x_um=180.0, corr_y_um=420.0,
                            contrast=choroid_contrast)
    return mottle, choroid
