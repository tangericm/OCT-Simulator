"""Ready-made sample presets: mirror, paper, cornea, macula.

Optical parameters (refractive index, backscatter, scattering coefficient,
layer thickness) are physically motivated but **illustrative** — they are the
knobs to calibrate against real scans, not measured constants. Backscatter
values are on a shared relative scale; the bright/dark *alternation* of the
retinal layers is the physically meaningful part, not the absolute numbers.

References for the ballpark values: retinal layer thicknesses from clinical
SD-OCT normative data; tissue group index ~1.38; cornea ~1.376; the retina's
characteristic hyper/hypo-reflective layer pattern (NFL/IPL/OPL/ELM/EZ/RPE
bright; GCL/INL/ONL dark). Treat every number here as a starting estimate.
"""

from __future__ import annotations

import numpy as np

from .sample import Layer, Sample


def mirror(depth_um: float = 400.0, reflectivity: float = 0.95) -> Sample:
    """A single specular reflector — reconstructs to one bright line.

    Use it to validate the axial PSF: the reconstructed peak width should match
    ``LightSource.axial_resolution_um`` (times the ~1.5x Hann broadening).
    """
    return Sample(
        name="mirror",
        top_n=1.0,
        surface_um=depth_um,
        layers=[
            Layer("mirror", thickness_um=20.0, n=1.5,
                  interface_reflectivity=reflectivity,
                  backscatter=0.0, density_per_um2=0.0),
        ],
    )


def paper(thickness_um: float = 300.0) -> Sample:
    """A dense turbid slab — bright surface, strong fully-developed speckle, and
    a steep Beer–Lambert roll-off. The canonical speckle-statistics target."""
    return Sample(
        name="paper",
        top_n=1.0,
        surface_um=150.0,
        layers=[
            Layer("paper", thickness_um=thickness_um, n=1.5,
                  backscatter=6.0e-2, mu_s_per_mm=25.0, density_per_um2=1.2),
        ],
    )


def cornea() -> Sample:
    """Anterior segment: bright anterior/posterior surfaces bracketing a nearly
    transparent (weakly scattering) stroma, with a gentle anterior curvature."""
    R_um = 7800.0                      # anterior corneal radius of curvature
    layers = [
        Layer("epithelium", thickness_um=50.0, n=1.401,
              backscatter=8.0e-3, mu_s_per_mm=2.0, density_per_um2=0.3),
        Layer("Bowman", thickness_um=17.0, n=1.380,
              backscatter=1.2e-2, mu_s_per_mm=3.0, density_per_um2=0.3),
        Layer("stroma", thickness_um=480.0, n=1.375,
              backscatter=4.0e-3, mu_s_per_mm=1.5, density_per_um2=0.25),
        # posterior interface (Descemet/endothelium -> aqueous) as a thin, bright
        # boundary; the explicit reflectivity gives the crisp posterior line.
        Layer("endothelium", thickness_um=10.0, n=1.336,
              interface_reflectivity=0.10,
              backscatter=6.0e-3, mu_s_per_mm=2.0, density_per_um2=0.3),
    ]

    def curvature(x: np.ndarray) -> np.ndarray:
        xc = x - x.mean()
        return (xc * xc) / (2.0 * R_um)     # shallow parabolic sag

    return Sample(name="cornea", layers=layers, top_n=1.0,
                  surface_um=150.0, curvature=curvature)


def macula() -> Sample:
    """Layered retina with the characteristic hyper/hypo-reflective pattern.

    Ordered vitreal -> scleral. A gentle contour stands in for the macular
    curvature; a true foveal pit (inner-layer thinning toward the centre) is a
    natural next extension of the fixed-thickness layer model.
    """
    # name, thickness_um, n, backscatter (relative), mu_s_per_mm, density
    spec = [
        ("NFL", 32.0, 1.38, 4.5e-2, 4.0, 0.5),   # bright
        ("GCL", 35.0, 1.38, 8.0e-3, 2.0, 0.3),   # dark
        ("IPL", 36.0, 1.38, 3.0e-2, 3.0, 0.4),   # bright-ish
        ("INL", 30.0, 1.38, 7.0e-3, 2.0, 0.3),   # dark
        ("OPL", 26.0, 1.38, 3.2e-2, 3.0, 0.4),   # bright
        ("ONL", 80.0, 1.38, 6.0e-3, 1.8, 0.3),   # dark
        ("ELM", 5.0, 1.38, 3.0e-2, 3.0, 0.4),    # thin bright line
        ("EZ", 10.0, 1.40, 8.0e-2, 5.0, 0.6),    # ellipsoid zone, very bright
        ("IZ", 10.0, 1.40, 5.0e-2, 4.0, 0.5),    # interdigitation, bright
        ("RPE", 15.0, 1.40, 9.0e-2, 6.0, 0.7),   # very bright / high scatter
        ("choroid", 200.0, 1.38, 2.5e-2, 3.0, 0.4),
    ]
    layers = [
        Layer(name, thickness_um=t, n=n, backscatter=b,
              mu_s_per_mm=mu, density_per_um2=d)
        for (name, t, n, b, mu, d) in spec
    ]

    def curvature(x: np.ndarray) -> np.ndarray:
        xc = x - x.mean()
        return (xc * xc) / (2.0 * 12000.0)     # very shallow macular contour

    return Sample(name="macula", layers=layers, top_n=1.336,
                  surface_um=140.0, curvature=curvature)


#: name -> zero-arg builder, for the gallery / CLI.
PRESETS = {
    "mirror": mirror,
    "paper": paper,
    "cornea": cornea,
    "macula": macula,
}
