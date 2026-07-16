"""Learn from UNLABELED OCT B-scans — texture, heterogeneity, and the layer-brightness
fingerprint — without any segmentation.

Segmentations tell us *where* the layers are; a large pile of plain B-scans tells us how the
retina actually *looks*: the relative brightness of the bands, the speckle contrast and grain,
the intra-layer heterogeneity, the noise floor, the total-thickness spread, and the curvature.
This module extracts all of that with a light, label-free ILM+RPE surface finder:

1. per-image robust normalization (display OCT is log-compressed 8-bit, often per-image
   contrast-stretched — only *relative* structure is comparable, so we normalize each image);
2. column-wise ILM (tissue onset) and RPE (brightest deep band) detection;
3. warp every A-line onto a normalised ILM→RPE depth axis and average → a **reflectivity
   template** whose peaks are the bright layers (NFL/IPL/OPL/EZ/RPE);
4. speckle contrast + axial/lateral grain (autocorrelation), the tissue intensity histogram,
   the vitreous noise floor, retinal thickness (px), and the RPE curvature.

:func:`layer_reflectivity_from_template` then reads per-layer relative reflectivity off the
template at each layer's fractional depth (from a shape model), giving **data-grounded optics
without segmentation**. Nothing is redistributed here — point the loader at any folder of OCT
images (Kermany OCT2017 *NORMAL*, Srinivasan 2014, OCTID, OCTDL, …).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

N_TEMPLATE = 120          # samples across the normalised ILM(0)→RPE(1)→+margin axis
_RPE_FRAC = 1.0           # RPE sits at 1.0 on the normalised axis; template runs to ~1.25


@dataclass
class TextureProfile:
    """Label-free appearance statistics learned from a pile of OCT B-scans."""

    reflectivity_template: list          # [N_TEMPLATE] mean normalised A-line, ILM→RPE→margin
    template_axis: list                  # [N_TEMPLATE] normalised depth (0=ILM, 1=RPE)
    speckle_contrast: float              # intensity std/mean within the retina
    heterogeneity: float                 # low-freq intra-band variation (mottle strength)
    axial_grain_um: float                # speckle autocorrelation FWHM, axial
    lateral_grain_um: float              # speckle autocorrelation FWHM, lateral
    noise_floor_frac: float              # vitreous background / retinal-peak brightness
    total_retina_um_mean: float
    total_retina_um_std: float
    curvature_concave_frac: float        # fraction of scans whose RPE is concave-up (a bowl)
    intensity_hist: list = field(default_factory=list)   # 32-bin normalised tissue histogram
    n_images: int = 0
    source: str = "unlabeled"

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(**json.load(fh))


#: bundled texture profiles fitted on public unlabeled datasets.
_BUNDLED_TEXTURE = {"srinivasan": "texture_srinivasan.json", "normal": "texture_srinivasan.json",
                    "default": "texture_srinivasan.json"}


def bundled_texture(name: str = "default") -> TextureProfile:
    """Load a bundled :class:`TextureProfile` (default: Srinivasan-2014 NORMAL, 1407 B-scans)."""
    fn = _BUNDLED_TEXTURE.get(name, name)
    if not os.path.isabs(fn):
        fn = os.path.join(os.path.dirname(__file__), fn)
    return TextureProfile.load(fn)


# --------------------------------------------------------------------------- io
def load_images(root: str, *, limit: int | None = None, exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")):
    """Yield grayscale float B-scans from a folder tree (recursively)."""
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, "**", "*" + e), recursive=True)
    paths = sorted(paths)
    if limit:
        paths = paths[:limit]
    for p in paths:
        img = _read_gray(p)
        if img is not None and img.ndim == 2 and min(img.shape) >= 32:
            yield img


def _read_gray(path):
    try:
        import tifffile
        if path.lower().endswith((".tif", ".tiff")):
            a = tifffile.imread(path)
            if a.ndim == 3:                    # a multi-frame volume -> yield handled by caller
                a = a[a.shape[0] // 2]         # take the central B-scan
            return a.astype(np.float64)
    except Exception:
        pass
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("L"), float)
    except Exception:
        return None


# --------------------------------------------------------------------------- surfaces
def _normalize(img):
    """Per-image robust [0,1] normalization (2nd–99.8th percentile)."""
    lo, hi = np.percentile(img, (2.0, 99.8))
    if hi <= lo:
        return None
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def _detect_ilm_rpe(img, smooth=7, axial_pitch_um=3.87, max_retina_um=520.0, border=10):
    """Column-wise ILM (tissue onset) and RPE (brightest outer-retinal band), or (None, None).

    The RPE search is capped at ``max_retina_um`` below the ILM so a bright image-border
    artifact (common in these display TIFFs) is never mistaken for the RPE; ``border`` rows are
    trimmed top and bottom first.
    """
    H, W = img.shape
    top, bot = border, H - border
    k = np.ones(smooth) / smooth
    col = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, img)
    max_th = int(max_retina_um / axial_pitch_um)
    ilm = np.full(W, -1); rpe = np.full(W, -1)
    for c in range(W):
        v = col[top:bot, c]
        thr = 0.35 * v.max()
        above = np.where(v > thr)[0]
        if above.size < 5:
            continue
        ilm[c] = above[0] + top
        lo = ilm[c] + max(12, int(0.5 * max_th))              # RPE is the deep outer band
        hi = min(ilm[c] + max_th, bot)
        if hi > lo + 2:
            rpe[c] = lo + int(np.argmax(col[lo:hi, c]))
    good = (ilm >= 0) & (rpe > ilm + 10)
    if good.sum() < 0.4 * W:
        return None, None
    # median-smooth the surfaces and interpolate the dropped columns
    xs = np.arange(W)
    ilm = np.interp(xs, xs[good], _medfilt(ilm[good].astype(float), 9))
    rpe = np.interp(xs, xs[good], _medfilt(rpe[good].astype(float), 9))
    return ilm, rpe


def _medfilt(v, k):
    k |= 1
    pad = k // 2
    vp = np.pad(v, pad, mode="edge")
    return np.array([np.median(vp[i:i + k]) for i in range(v.size)])


# --------------------------------------------------------------------------- extraction
def extract_texture(images, *, axial_pitch_um=3.87, lateral_pitch_um=6.0,
                    source="unlabeled") -> TextureProfile:
    """Aggregate a :class:`TextureProfile` over an iterable of grayscale B-scans."""
    axis = np.linspace(0.0, 1.25, N_TEMPLATE)
    templates, contrasts, hetero, thick, noise, concave = [], [], [], [], [], []
    ax_grain, lat_grain, hists = [], [], []
    n = 0
    for raw in images:
        img = _normalize(raw)
        if img is None:
            continue
        ilm, rpe = _detect_ilm_rpe(img, axial_pitch_um=axial_pitch_um)
        if ilm is None:
            continue
        H, W = img.shape
        # 1) warp each column onto the normalised ILM->RPE axis and average
        tpl = np.zeros(N_TEMPLATE); m = 0
        band_vals = []
        for c in range(0, W, 2):
            a, b = ilm[c], rpe[c]
            if b - a < 12:
                continue
            src = a + axis * (b - a)                       # normalised depth -> row
            tpl += np.interp(src, np.arange(H), img[:, c])
            m += 1
            band_vals.append(img[int(a):int(b), c])
        if m < 0.2 * W:
            continue
        tpl /= m
        templates.append(tpl)
        # 2) speckle contrast + heterogeneity within the retina
        band = np.concatenate(band_vals)
        band = band[band > 0.05]
        if band.size:
            contrasts.append(float(np.std(band) / (np.mean(band) + 1e-9)))
            hists.append(np.histogram(band, bins=32, range=(0, 1), density=True)[0])
        # 3) thickness + curvature + noise
        thick.append(float(np.median(rpe - ilm) * axial_pitch_um))
        xc = np.arange(W) - W / 2
        a2 = np.polyfit(xc, rpe, 2)[0]                     # RPE parabola curvature
        concave.append(1.0 if a2 < 0 else 0.0)            # rows increase downward: a<0 = bowl
        vit = img[:int(np.median(ilm)) - 5] if np.median(ilm) > 8 else img[:3]
        noise.append(float(np.median(vit)) if vit.size else 0.0)
        # 4) speckle grain via autocorrelation of a mid-retina patch
        ax_g, lat_g = _grain(img, ilm, rpe, axial_pitch_um, lateral_pitch_um)
        if ax_g:
            ax_grain.append(ax_g); lat_grain.append(lat_g)
        n += 1

    if not templates:
        raise RuntimeError("no B-scans yielded a usable retina; check the image folder")
    tpl = np.median(np.array(templates), axis=0)
    tpl = tpl / (tpl.max() + 1e-9)
    het = _heterogeneity(np.array(templates))
    return TextureProfile(
        reflectivity_template=tpl.tolist(), template_axis=axis.tolist(),
        speckle_contrast=float(np.median(contrasts)) if contrasts else 0.5,
        heterogeneity=float(het),
        axial_grain_um=float(np.median(ax_grain)) if ax_grain else axial_pitch_um * 2,
        lateral_grain_um=float(np.median(lat_grain)) if lat_grain else lateral_pitch_um * 2,
        noise_floor_frac=float(np.median(noise)) if noise else 0.05,
        total_retina_um_mean=float(np.mean(thick)), total_retina_um_std=float(np.std(thick)),
        curvature_concave_frac=float(np.mean(concave)) if concave else 0.5,
        intensity_hist=(np.median(np.array(hists), axis=0).tolist() if hists else []),
        n_images=n, source=f"{source} ({n} B-scans)")


def _heterogeneity(templates):
    """Between-scan spread of the reflectivity template (band-brightness variability)."""
    t = templates / (templates.max(axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.std(t, axis=0)))


def _grain(img, ilm, rpe, dz, dx):
    """Axial/lateral autocorrelation FWHM (µm) of a mid-retina patch."""
    c0 = img.shape[1] // 2
    a = int(np.median(ilm)); b = int(np.median(rpe))
    if b - a < 20 or c0 < 40:
        return None, None
    patch = img[a + 5:b - 5, c0 - 30:c0 + 30]
    if patch.size < 100:
        return None, None
    patch = patch - patch.mean()
    az = _acf_fwhm(patch.mean(axis=1)) * dz
    lat = _acf_fwhm(patch.mean(axis=0)) * dx
    return az, lat


def _acf_fwhm(v):
    v = v - v.mean()
    ac = np.correlate(v, v, "full")[v.size - 1:]
    ac = ac / (ac[0] + 1e-12)
    half = np.where(ac < 0.5)[0]
    return float(half[0]) if half.size else 1.0


# --------------------------------------------------------------------------- apply
def layer_reflectivity_from_template(profile: TextureProfile, layer_names, layer_thickness_um):
    """Relative per-layer reflectivity (amplitude) read off the template at each layer's
    fractional depth — data-grounded optics WITHOUT segmentation.

    ``layer_thickness_um`` is the mean neurosensory thickness per layer (e.g. a shape model's
    ``mean_thickness`` at the parafovea); the template (ILM=0 → RPE=1) is sampled at each
    layer's mid-depth fraction. Choroid (beyond RPE) is left to the caller.
    """
    axis = np.asarray(profile.template_axis)
    tpl = np.asarray(profile.reflectivity_template)
    neuro = [(nm, t) for nm, t in zip(layer_names, layer_thickness_um) if nm != "choroid"]
    total = sum(t for _, t in neuro) or 1.0
    out, cum = {}, 0.0
    for nm, t in neuro:
        frac = (cum + 0.5 * t) / total          # mid-depth fraction ILM(0)->RPE(1)
        out[nm] = float(np.sqrt(max(np.interp(frac, axis, tpl), 1e-4)))  # dB-ish -> amplitude
        cum += t
    norm = np.median(list(out.values())) or 1.0
    return {k: v / norm for k, v in out.items()}
