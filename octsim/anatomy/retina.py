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

from ..features import BloodVessel, ChoroidVasculature, LaminaCribrosa
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

#: Axial reflectivity-transition width for the photoreceptor bands and the RPE. These
#: interfaces are physically sub-micron, unlike the genuinely gradual inner-retinal ones,
#: so they get their own width instead of the global ``boundary_sharpness_um``. Kept just
#: above zero so the map is still band-limited relative to the depth pixel.
OUTER_SHARPNESS_UM = 1.5

#: Optic-cup excavation depth range (optical µm). Measured on the Maestro3 disc scans
#: against the peripapillary rim: ~720 µm at both 6 and 9 mm. Shared by every RetinaModel
#: builder so the three constructors cannot drift apart again.
ONH_CUP_DEPTH_RANGE = (560.0, 900.0)

#: Peripapillary RNFL thickening. Ganglion-cell axons converge on the canal, so the nerve
#: fibre layer thins toward the fovea and thickens toward the disc — gradually, over
#: millimetres, reaching ~3x its parafoveal thickness at the margin.
#:
#: The model previously expressed this only as a 280 µm-wide rim term inside the ONH
#: builder, so simulated NFL measured a flat 27-38 µm across an entire 6 or 9 mm B-scan
#: with a single 91 µm spike at the disc. Worse, the *macular* cross-section never received
#: the disc position at all, so a wide macular scan pointed at the disc had no nasal
#: thickening whatsoever.
NFL_DISC_GAIN = 2.4          # peak multiplier at the disc margin
NFL_DISC_DECAY_UM = 1200.0   # e-folding distance of the approach


def nfl_disc_gradient(x, y_um, disc_x, disc_y_um, bmo_um):
    """Multiplier on NFL thickness from proximity to the optic disc (1 = far away)."""
    if disc_x is None:
        return 1.0
    r_disc = np.sqrt((np.asarray(x, float) - float(disc_x)) ** 2
                     + (float(y_um) - float(disc_y_um)) ** 2)
    reach = np.clip(r_disc - float(bmo_um), 0.0, None)
    return 1.0 + NFL_DISC_GAIN * np.exp(-reach / NFL_DISC_DECAY_UM)


#: Choroidal sublayer geometry, as a fraction of choroid thickness (Bruch's -> sclera), with
#: lumen calibre ranges in µm. Sattler's layer holds medium vessels, Haller's the large ones;
#: the choriocapillaris is too fine to resolve laterally and is rendered as a bright band
#: rather than as lumens. See Pichi et al., Indian J Ophthalmol 2018;66(12):1716-26.
CHOROID_SUBLAYERS = (
    # (name,     z_rel range,   rx µm,        rz µm,       per-mm density)
    ("sattler",  (0.12, 0.48), (12.0, 36.0),  (12.0, 30.0), 17.0),
    ("haller",   (0.45, 0.95), (36.0, 105.0), (34.0, 85.0), 6.5),
)


#: Fallback choroidal bed for models that do not supply one (see ``RetinaModel.__post_init__``).
#: The lumen COUNT scales with the span, so drawing over a wide field keeps the per-mm density
#: correct at any frame width; the field is only made wide enough to cover the largest scan.
_FALLBACK_LUMEN_FOV_UM = 12000.0
_FALLBACK_LUMEN_SEED = 0x0C40


def random_choroid_lumens(rng, *, lateral_fov_um: float, y_span_um: float = 1400.0) -> list:
    """Draw a 3-D population of choroidal vessel lumens for one eye.

    Held on the :class:`RetinaModel` and cross-sectioned per frame, so a volume sees one
    coherent vascular bed rather than an independent pattern per B-scan.
    """
    from ..features import ChoroidLumen
    lumens = []
    span_mm = lateral_fov_um / 1000.0
    for _name, (z0, z1), (rx0, rx1), (rz0, rz1), per_mm in CHOROID_SUBLAYERS:
        n = int(round(per_mm * span_mm * (1.0 + 2.0 * y_span_um / 1000.0)))
        for _ in range(max(n, 0)):
            rx = float(rng.uniform(rx0, rx1))
            lumens.append(ChoroidLumen(
                x0_um=float(rng.uniform(-0.05, 1.05)) * lateral_fov_um,
                y0_um=float(rng.uniform(-y_span_um, y_span_um)),
                z_rel=float(rng.uniform(z0, z1)),
                rx_um=rx,
                # vessels run mostly in-plane, so they are far longer than they are wide
                ry_um=rx * float(rng.uniform(3.0, 12.0)),
                rz_um=float(rng.uniform(rz0, rz1)),
                gain=float(rng.uniform(0.22, 0.52))))
    return lumens


def random_macular_vessels(rng, *, center_x_um: float, lateral_fov_um: float,
                           location: str = "macula") -> list:
    """Draw a physiological set of retinal vessels for a scene (shared by both the
    parametric and shape-model retina builders — one place so the FAZ and shadow tuning
    can't drift between them).

    Calibrated to **frame-averaged clinical B-scans** (OCTA-500 etc.): superficial vessels
    ring the foveal avascular zone and cast only a *subtle* posterior shadow — the large dark
    lumina / heavy shadows of a single-frame render are barely visible in acquired data, so
    the caliber and shadow depth are kept modest.
    """
    from ..features import BloodVessel
    vessels = []
    for _ in range(int(rng.integers(2, 6))):
        vx = float(rng.uniform(0.08, 0.92)) * lateral_fov_um
        if location == "macula" and abs(vx - center_x_um) < FAZ_RADIUS_UM:
            # ring the fovea rather than cross it: land just outside the avascular zone
            vx = center_x_um + np.sign(vx - center_x_um + 1e-9) * (
                FAZ_RADIUS_UM + float(rng.uniform(10.0, 90.0)))
        cal = float(rng.uniform(25.0, 95.0))          # fewer huge trunks -> less blobby
        plexus = 0.0 if cal > 75.0 else float(rng.choice([0.0, 0.0, 25.0, 50.0]))
        vessels.append(BloodVessel(
            x0_um=vx, y0_um=float(rng.uniform(-200, 200)), caliber_um=cal,
            plexus_offset_um=plexus + float(rng.uniform(0.0, 8.0)),
            slope=float(rng.uniform(-0.4, 0.4)),
            shadow_scale=float(rng.uniform(0.8, 1.2))))   # physical blood attenuation
    return vessels

# Vitreal -> scleral. Thicknesses ~ clinical parafoveal norms (µm); the bright/dark
# alternation (NFL/IPL/OPL/EZ/RPE bright; GCL/INL/ONL dark) is the meaningful part.
# Contrast widened and the outer complex brightened to match clinical B-scans.
RETINA_LAYERS: list[RetinaLayer] = [
    # Backscatter set by matching the ILM-to-choroid mean A-line profile against the
    # Maestro3 corpus (6 frames, scripts/validate_realism.py), band by band, with every
    # value expressed relative to the outer-complex peak. Two things the previous table got
    # badly wrong, both visible as "the layers look like stripes":
    #   * the NFL was ~22 dB too bright relative to the outer complex — bright enough that
    #     763 of 1024 columns had their per-column maximum in the NFL rather than the EZ,
    #     which no real B-scan does;
    #   * the bright/dark alternation within the inner retina was ~7 dB where real macular
    #     scans show ~3 dB, so GCL and INL read as near-black gaps between bright bands
    #     instead of a continuously textured plateau.
    RetinaLayer("NFL", 30.0, 1.38, 8.2e-3, 4.5, "nfl", interface_reflectivity=3.0e-3),
    RetinaLayer("GCL", 38.0, 1.38, 7.5e-3, 1.6, "inner"),
    RetinaLayer("IPL", 40.0, 1.38, 1.26e-2, 3.0, "inner"),
    RetinaLayer("INL", 30.0, 1.38, 6.2e-3, 1.5, "inner"),
    RetinaLayer("OPL", 25.0, 1.38, 1.35e-2, 3.0, "inner"),
    RetinaLayer("ONL", 78.0, 1.38, 7.2e-3, 1.2, "onl"),
    RetinaLayer("ELM", 5.0, 1.38, 3.5e-2, 3.0, "photoreceptor",
                interface_reflectivity=0.012),
    # Outer-complex levels measured on the Maestro3 corpus (octsim/data/... 6mm_1024Aline*)
    # from the RPE-flattened mean A-line over a window wide enough to contain the whole
    # complex: the RPE/BM maximum sits ~3 dB ABOVE the EZ maximum, with a ~9 dB trough
    # between them and ~35 um of optical separation. (An earlier pass here used a narrow
    # window that clipped the RPE peak and concluded the opposite; the wide-window numbers
    # are the correct ones.) Both peaks must be present and separated — a single fat bright
    # band is the most conspicuous way a simulated outer retina reads as fake.
    RetinaLayer("EZ", 9.0, 1.41, 1.55e-1, 6.0, "photoreceptor"),
    # Outer segments: a genuinely LOW-scattering band. Without it the outer complex renders
    # as one fat spike, because EZ and RPE are separated only by the bright IZ. Real B-scans
    # show two maxima ~25 µm of tissue apart with a 4-8 dB trough between them, and this is
    # the trough. OCTA-500's "OS" segmentation layer (8.5-11.5 µm) is exactly this band —
    # it used to be mapped onto IZ, which is why the trough was being filled in.
    RetinaLayer("OS", 24.0, 1.41, 3.2e-2, 1.8, "photoreceptor"),
    RetinaLayer("IZ", 4.0, 1.41, 8.5e-2, 5.0, "photoreceptor"),
    # RPE backscatter is set from the same measurement.
    RetinaLayer("RPE", 18.0, 1.42, 1.85e-1, 8.0, "rpe", water_fraction=0.5, melanin=1.0),
    # Bruch's membrane + choriocapillaris, as one resolvable band. Real B-scans show the
    # outer complex holding near its peak for ~25 um of tissue BELOW the RPE before falling
    # into the choroid; that posterior plateau is BM+CC. Modelling it as a brightness gain
    # on the choroid could not work — it would need a ~36x gain on a layer 25 dB dimmer than
    # the RPE, which is just a layer wearing a disguise.
    RetinaLayer("BMCC", 16.0, 1.40, 9.2e-2, 5.0, "rpe", water_fraction=0.6, melanin=0.2),
    RetinaLayer("choroid", 260.0, 1.38, 3.0e-3, 3.0, "choroid", water_fraction=0.6, melanin=0.5),
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
    layer_sharpness_um: object = None    # [L] per-layer transition width (or None = global)
    speckle_decorrelation: float = 0.0   # fraction of speckle power that is laterally white
    ms_fraction: float = 0.0             # multiple-scattering pedestal strength (0 = off)
    ms_length_um: float = 180.0          # pedestal decay length
    texture_field: object = None         # CoherentField for volume-coherent mottling (or None)
    speckle_shape: float | None = None   # K-dist effective #scatterers M (None = Rayleigh)


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
    onh_cup_depth_um: float = 700.0   # optical um; real Maestro3 discs measure ~720
    #: central retinal vessel trunk (caliber and its offset from the canal axis)
    disc_trunk_caliber_um: float = 240.0
    disc_trunk_offset_um: float = -90.0
    disc_dx_um: float = 4760.0            # signed fovea->disc lateral offset (temporal fovea)
    disc_present: bool = True             # include the disc so it appears at a wide FOV
    disc_fill_backscatter: float = 2.2e-2  # neural-canal / prelaminar tissue reflectivity
    vessels: list = field(default_factory=list)
    # global curvature: concave-up posterior-pole bowl (center deepest), saturating so
    # a wide scan stays in-window. `curvature_radius_um` sets the near-fovea radius.
    curvature_radius_um: float = 12000.0  # apex radius of curvature (~globe radius)
    curvature_sag_um: float = 1800.0      # depth-window overflow guard (not a shape knob)
    #: Distance from the scan pivot (near the pupil) to the retina, µm. Drives the fan-beam
    #: sag term in :meth:`_contour`. This is a SCAN-GEOMETRY quantity, not anatomy, so it is
    #: kept separate from ``curvature_radius_um`` (which stays at its literature value).
    pivot_distance_um: float = 20000.0
    scan_distortion_correction: float = 0.86
    curvature_conic: float = 0.0          # conic constant K (0=sphere, <-1 flattens periphery)
    curvature_concave: bool = True        # True = bowl (real posterior pole); False = legacy dome
    boundary_sharpness_um: float = 8.0    # axial reflectivity-transition width (softens bands)
    choroid_contrast: float = 0.32        # choroidal vascular lucency depth (subtle, averaged look)
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
    choroid_lumens: object = None        # list[ChoroidLumen]: the 3-D choroidal vessel bed
    # non-Rayleigh volume speckle: effective #scatterers per resolution cell (K-distribution
    # shape M). None = fully-developed Rayleigh (legacy). Small M => super-Rayleigh heavy tail.
    speckle_shape: float | None = None
    #: Fraction of speckle power decorrelated per A-line (see the note in
    #: build_reflectivity_from_boundaries). Fitted to the 2048-A-line lateral grain.
    speckle_decorrelation: float = 0.12
    #: Multiple-scattering diffuse pedestal (see build_reflectivity_from_boundaries).
    #: Fitted against the real mean A-line tail below the RPE: without it the simulated
    #: choroid falls ~6 dB too fast over the first 400 um. Part of the choroidal signal is
    #: therefore multiply-scattered light rather than choroidal backscatter, which is why
    #: the choroid's own backscatter was lowered when this was switched on.
    ms_fraction: float = 0.025
    ms_length_um: float = 180.0

    def __post_init__(self):
        # The choroid used to be a smooth random field (`ChoroidTexture`), which generated
        # its own noise when no explicit field was supplied. `ChoroidVasculature` is a list
        # of discrete lumens instead, so an unset list is not "generate one" but "no vessels
        # at all" — a flat choroid on every path that does not populate it: `default_retina`
        # (previews/tests), `retina_from_profile` (the calibrated-profile path), and direct
        # construction. Only `random_retina` and `retina_from_shape_model` set it explicitly.
        #
        # So fall back to a bed here rather than at the call sites, which also covers models
        # built by hand. Seeded, because previews and tests must stay reproducible.
        if self.choroid_lumens is None:
            self.choroid_lumens = random_choroid_lumens(
                np.random.default_rng(_FALLBACK_LUMEN_SEED),
                lateral_fov_um=_FALLBACK_LUMEN_FOV_UM)

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
        cross.layer_sharpness_um = np.array(
            [OUTER_SHARPNESS_UM if g in ("photoreceptor", "rpe") else self.boundary_sharpness_um
             for g in groups], float)
        cross.speckle_decorrelation = self.speckle_decorrelation
        cross.ms_fraction = self.ms_fraction
        cross.ms_length_um = self.ms_length_um
        cross.texture_field = self.mottle_field
        cross.speckle_shape = self.speckle_shape
        cross.layer_directional = self._directional_gain(cross.opt_b, x, groups)
        return cross

    @classmethod
    def _onh_groups(cls, n: int) -> list:
        by_name = {ly.name: ly.group for ly in RETINA_LAYERS}
        return [by_name.get(name, "canal") for name in cls._FIELD_ORDER[:n]]

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
        R = self.curvature_radius_um              # apex radius of curvature (µm)
        if R <= 0:
            bowl = np.zeros_like(x, dtype=float)
        else:
            # CONIC (aspheric) sag: z = (h²/R) / (1 + sqrt(1 - (1+K)(h/R)²)). The conic constant K
            # (``curvature_conic``) sets peripheral shape at fixed apex radius R: K=0 = sphere,
            # K=-1 = parabola, K<-1 = hyperbolic FLATTENING toward the periphery, K>0 = oblate steepening.
            # Real posterior-pole B-scans are ~spherical near the fovea but FLATTER at wide FOV than a
            # sphere (device distortion-correction + aspheric anatomy — deep-research), which a single
            # radius could not capture: a pure cap steepens at the edge and over-curves 9-12 mm scans.
            # K=0 reproduces the previous spherical cap exactly (label-map invariant at the anchor).
            K = float(getattr(self, "curvature_conic", 0.0))
            h = np.abs(xc)
            disc = 1.0 - (1.0 + K) * (h / R) ** 2
            disc = np.maximum(disc, 1e-6)              # clamp (past the conic's real domain -> flat asymptote)
            bowl = (h * h / R) / (1.0 + np.sqrt(disc))
        # FAN-BEAM term. Most of the curvature on a B-scan is not anatomy: the beam pivots
        # near the pupil, so an A-line at lateral offset h travels an extra optical path
        # p·(1 - cos(h/p)) ≈ h²/(2p) relative to the centre. Anatomy alone (R = 12 mm) gives
        # h²/(2R) = 375 µm of sag over a 6 mm scan, but the measured Maestro3 sag is ~600 µm;
        # closing that with the anatomic radius would require a 6.8 mm eye. This is the term
        # docs/CURVATURE_FALLOFF.md identified and never built, and it is why a single fitted
        # R could never be right at two different FOVs at once — the optimiser was using an
        # anatomy knob to absorb a scan-geometry artifact.
        p = float(getattr(self, "pivot_distance_um", 0.0) or 0.0)
        if p > 0:
            bowl = bowl + p * (1.0 - np.cos(np.clip(np.abs(xc) / p, 0.0, np.pi / 2)))
        # Depth-window overflow guard, applied AFTER both sag terms (clamping only the
        # anatomic bowl would let the fan-beam term escape the guard and push a wide scan out
        # of the axial window). A hard np.minimum is the wrong shape for a guard: it creates
        # a genuinely flat plateau wherever it engages, and a flat boundary has zero gradient,
        # which silently switches off the angle-dependent (Stiles-Crawford) reflectance that
        # is derived from the boundary slope. This soft saturation is monotone — within 0.1 %
        # of the identity until ~half the limit, and asymptotic to it — so depth stays bounded
        # while the periphery keeps a real slope.
        if self.curvature_sag_um > 0:
            s = float(self.curvature_sag_um)
            bowl = bowl / np.power(1.0 + np.power(bowl / s, 6.0), 1.0 / 6.0)
        # Vendor scan-distortion correction: devices flatten the geometric sag by differing
        # amounts (Maestro3 ~0.86, Maestro2 ~0.43), so this is a device parameter supplied by
        # the config, not an anatomic one.
        bowl = bowl * float(getattr(self, "scan_distortion_correction", 1.0))
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

        # Peripapillary RNFL thickening. The shape model is fovea-centred and carries no
        # disc, so without this a 9 mm macular scan reaching toward the optic nerve shows a
        # flat NFL where the real one is 3x thicker nasally.
        _fovea_x, disc_x = self._landmarks()
        if disc_x is not None:
            g = nfl_disc_gradient(x, y_um, disc_x, self.center_y_um, self.onh_radius_um)
            for i, ly in enumerate(layers):
                if ly.group == "nfl":
                    t_phys[i] = t_phys[i] * g

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
            cross.features.append(ChoroidVasculature(
                top_opt=cross.opt_b[i], bot_opt=cross.opt_b[i + 1],
                lumens=(self.choroid_lumens or [])))
        return cross

    _FIELD_ORDER = ["NFL", "canal", "GCL", "IPL", "INL", "OPL", "ONL",
                    "ELM", "EZ", "OS", "IZ", "RPE", "BMCC", "choroid"]

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
        # Prelaminar/laminar tissue attenuates less than the dense neural retina, which is
        # why a real disc still shows signal ~650 um below the cup floor.
        canal = RetinaLayer("canal", 0.0, 1.38, self.disc_fill_backscatter, 1.6, "canal")
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
            # Termination ramp. Measured real disc margins are STEEP — close to vertical
            # over ~150 µm — so the ramp is deliberately short. (The staircase artifact this
            # once produced was never about steepness: it came from every boundary collapsing
            # into one depth bin and its Fresnel lines summing coherently. That is a
            # sub-pixel-deposition problem, and the interfaces are now dim enough not to
            # stack into a visible cliff.)
            term = _smoothstep(bmo - 100.0, bmo + 50.0, r_disc)    # 1 outside, 0 inside
            rim = 1.0 - _smoothstep(bmo, bmo + 280.0, r_disc)
            # Steep-walled cup with a flatter floor, as measured: excavation is near-full
            # out to ~half the BMO radius, then climbs sharply to the rim.
            cup = self.onh_cup_depth_um * (1.0 - _smoothstep(0.12 * bmo, 0.88 * bmo, r_disc))
            for nm in neuro:
                t_final[nm] = t_mac[nm] * term                     # terminate at the BMO
            # The NFL is the one layer that does NOT terminate at the BMO: the nerve fibres
            # are what pass through the canal. It thins toward the macula and thickens
            # peripapillarly, then turns down into the canal — so it must stay present at
            # and inside the disc margin. Multiplying it by `term` like the other layers
            # deleted it over ~110 columns of a disc scan, leaving the rim bare.
            nfl_canal = 0.35 + 0.65 * _smoothstep(0.20 * bmo, bmo, r_disc)
            grad = nfl_disc_gradient(x, y_um, disc_x, self.center_y_um, bmo)
            t_final["NFL"] = t_mac["NFL"] * grad * nfl_canal
            # Canal fill depth. Real disc scans show signal ~650-700 um below the cup ILM
            # (prelaminar tissue + lamina); 260 um terminated the disc far too shallow.
            t_final["canal"] = 620.0 * (1.0 - term)
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
        cross.features.append(ChoroidVasculature(
            top_opt=cross.opt_b[ci], bot_opt=cross.opt_b[ci + 1],
            lumens=(self.choroid_lumens or [])))
        if disc_x is not None:
            kal = order.index("canal")
            cross.features.append(LaminaCribrosa(
                disc_x_um=float(disc_x), disc_y_um=float(self.center_y_um),
                radius_um=float(self.onh_radius_um), top_opt=cross.opt_b[kal]))
            # Central retinal vessel trunk: the largest single feature of a real disc B-scan,
            # and the source of the deep shadow through the canal. Emerges at the cup floor.
            cross.features.append(BloodVessel(
                x0_um=float(disc_x) + self.disc_trunk_offset_um, y0_um=float(self.center_y_um),
                caliber_um=self.disc_trunk_caliber_um, y_half_extent_um=self.onh_radius_um,
                plexus_offset_um=40.0))
        return cross


# --------------------------------------------------------------------------- builders
def default_retina(*, location: str = "macula", surface_opt_um: float = 200.0,
                   center_x_um: float = 500.0, lateral_fov_um: float = 1000.0) -> RetinaModel:
    """A fixed, reasonable retina (no randomness) for previews/tests."""
    vessels = [
        BloodVessel(x0_um=center_x_um - 300.0, y0_um=0.0, caliber_um=70.0),
        BloodVessel(x0_um=center_x_um + 320.0, y0_um=0.0, caliber_um=55.0,
                    plexus_offset_um=25.0),
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

    vessels = random_macular_vessels(rng, center_x_um=center_x,
                                     lateral_fov_um=lateral_fov_um, location=location)

    undulation = BoundaryUndulation.make(
        len(RETINA_LAYERS) + 2, rng,
        amp_um=float(rng.uniform(2.0, 6.0)), corr_um=float(rng.uniform(350.0, 600.0)))
    texture_contrast = float(rng.uniform(0.16, 0.30))
    choroid_contrast = float(rng.uniform(0.22, 0.36))     # subtle choroidal lucencies (averaged look)
    mottle_field, _ = _texture_fields(rng, texture_contrast, choroid_contrast)
    choroid_lumens = random_choroid_lumens(rng, lateral_fov_um=lateral_fov_um)

    return RetinaModel(
        location=location, layers=layers, surface_opt_um=surface_opt_um, center_x_um=center_x,
        center_y_um=float(rng.uniform(-100, 100)),
        pit_radius_um=float(rng.uniform(550.0, 850.0)),
        onl_bulge=float(rng.uniform(0.4, 0.7)),
        curvature_radius_um=float(rng.uniform(11000.0, 14000.0)),
        curvature_sag_um=float(rng.uniform(1700.0, 1900.0)),
        boundary_sharpness_um=float(rng.uniform(7.0, 11.0)),   # softer inter-layer transitions
        choroid_contrast=choroid_contrast,
        onh_radius_um=float(rng.uniform(620.0, 820.0)),
        onh_cup_depth_um=float(rng.uniform(*ONH_CUP_DEPTH_RANGE)),
        disc_dx_um=float(rng.choice([-1.0, 1.0]) * rng.uniform(4300.0, 5200.0)),
        tilt_um_per_mm=float(rng.uniform(-25.0, 25.0)),
        pit_asymmetry=float(rng.uniform(-0.5, 0.5)),
        undulation=undulation,
        texture_contrast=texture_contrast,
        interface_jitter=float(rng.uniform(0.20, 0.40)),
        vitreous_backscatter=float(rng.uniform(3.0e-4, 7.0e-4)),
        directional_strength=float(rng.uniform(0.45, 0.75)),
        henle_sign=float(rng.choice([-1.0, 0.0, 1.0]) * rng.uniform(0.5, 1.0)),
        mottle_field=mottle_field, choroid_lumens=choroid_lumens,
        vessels=vessels)


def _texture_fields(rng, texture_contrast, choroid_contrast):
    """Build the two volume-coherent 3-D texture fields (mottling + choroid) for a scene.

    ``k=32`` random-frequency components (up from 8): with the random-sign wavevectors in
    :class:`CoherentField` a few components read as a plaid/weave, but ~32 fill the spectrum into a
    natural isotropic mottle — the clinically-plausible spongy choroid instead of a diagonal grating.
    """
    from ..sample import CoherentField
    mottle = CoherentField(rng, k=32, corr_z_um=18.0, corr_x_um=70.0, corr_y_um=300.0,
                           contrast=texture_contrast)
    choroid = CoherentField(rng, k=32, corr_z_um=90.0, corr_x_um=180.0, corr_y_um=420.0,
                            contrast=choroid_contrast)
    return mottle, choroid
