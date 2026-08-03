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
    #: un-normalised inter-layer contrast — dB range of the column-averaged (speckle-reduced)
    #: ILM→RPE profile. The percentile-normalised stats above are contrast-BLIND, so this is the
    #: only statistic that pins the layer brightness DYNAMIC RANGE (keeps the sim from over-boosting
    #: layer contrast). 0.0 = not measured (older bundled profiles); the distance skips it then.
    retinal_contrast_db: float = 0.0
    #: normalized depth-intensity envelope: mean linear intensity in 8 slabs from the ILM down into
    #: the choroid, peak-normalized so the SHAPE (sensitivity roll-off + deep tissue attenuation)
    #: survives. The other stats stop at the RPE and are percentile-normalized, so they cannot see the
    #: deep fade. [] = not measured (older profiles); the falloff distance term skips it then.
    depth_intensity_env: list = field(default_factory=list)
    #: FOV-normalized RPE curvature coefficient (µm sag per (mm half-width)²) — the bowl steepness,
    #: comparable across fields-of-view. 0.0 = not measured; the curvature distance term skips it then.
    rpe_curvature_coef: float = 0.0
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


#: bundled texture profiles fitted on public unlabeled datasets. The default is the healthy
#: multi-dataset pool (Srinivasan + OCTDL + OCTID-normal + OCT-C8-normal); the single normal
#: datasets and the DISEASE categories (macular hole / CSR / AMD / CNV / DME / diabetic
#: retinopathy / drusen — pooled OCT-C8 + OCTID where both cover the class) are kept too. The
#: disease profiles carry realistic texture but distorted geometry, so they are NOT in the
#: healthy default (use them explicitly to flavour degraded/pathological corpora).
#: ``octa500`` is a NORMAL profile but stays separate: its RTVue-XR structural B-scans are
#: frame-AVERAGED (speckle contrast ~0.34, about 2.3 effective frames), a different acquisition
#: regime from the single-frame (~0.5) datasets that make up the default pool.
_BUNDLED_TEXTURE = {"default": "texture_pooled.json", "normal": "texture_pooled.json",
                    "pooled": "texture_pooled.json", "srinivasan": "texture_srinivasan.json",
                    "octdl": "texture_octdl.json", "octid": "texture_octid_normal.json",
                    "octc8": "texture_octc8.json", "octa500": "texture_octa500.json",
                    "mh": "texture_mh.json", "csr": "texture_csr.json",
                    "amd": "texture_amd.json", "dr": "texture_dr.json",
                    "cnv": "texture_cnv.json", "dme": "texture_dme.json",
                    "drusen": "texture_drusen.json"}


def bundled_texture(name: str = "default") -> TextureProfile:
    """Load a bundled :class:`TextureProfile` (default: the pooled multi-dataset NORMAL prior)."""
    fn = _BUNDLED_TEXTURE.get(name, name)
    if not os.path.isabs(fn):
        # bare corpus keywords (device_a_macula, …) live as texture_<name>.json
        cand = os.path.join(os.path.dirname(__file__), fn)
        if not os.path.isfile(cand) and name not in _BUNDLED_TEXTURE:
            alt = os.path.join(os.path.dirname(__file__), f"texture_{name}.json")
            if os.path.isfile(alt):
                cand = alt
        fn = cand
    return TextureProfile.load(fn)


def bundled_texture_meta(name: str) -> dict | None:
    """Load the sidecar ``texture_<name>_meta.json`` if present (alines, pitches, SNR).

    Real device targets store acquisition geometry here — not on
    :class:`TextureProfile` — so calibration renders can match the corpus A-line
    pitch. Returns ``None`` when no sidecar exists (pooled / clinical priors).
    """
    import json
    d = os.path.dirname(__file__)
    # bundled alias → texture_*.json stem; raw corpus names already match the stem
    fn = _BUNDLED_TEXTURE.get(name)
    if fn is not None:
        stem = os.path.splitext(fn)[0]
    else:
        stem = f"texture_{name}"
    path = os.path.join(d, f"{stem}_meta.json")
    if not os.path.isfile(path):
        # also accept texture_<name>_meta.json when name was already a full stem
        alt = os.path.join(d, f"texture_{name}_meta.json")
        path = alt if os.path.isfile(alt) else path
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pool_textures(profiles) -> TextureProfile:
    """Pool several :class:`TextureProfile` (from different datasets) into one, weighting each
    field by that dataset's B-scan count. The reflectivity templates share a fixed normalised
    axis, so they average directly. Use this to build a broad multi-dataset NORMAL prior."""
    profiles = [p for p in profiles if p and p.n_images > 0]
    if not profiles:
        raise ValueError("no profiles to pool")
    w = np.array([p.n_images for p in profiles], float)
    w = w / w.sum()
    wavg = lambda a: float(np.dot(w, [getattr(p, a) for p in profiles]))
    tpl = np.average(np.array([p.reflectivity_template for p in profiles]), axis=0, weights=w)
    hists = [p.intensity_hist for p in profiles if len(p.intensity_hist)]
    return TextureProfile(
        reflectivity_template=tpl.tolist(), template_axis=list(profiles[0].template_axis),
        speckle_contrast=wavg("speckle_contrast"), heterogeneity=wavg("heterogeneity"),
        axial_grain_um=wavg("axial_grain_um"), lateral_grain_um=wavg("lateral_grain_um"),
        noise_floor_frac=wavg("noise_floor_frac"),
        total_retina_um_mean=wavg("total_retina_um_mean"),
        total_retina_um_std=wavg("total_retina_um_std"),
        curvature_concave_frac=wavg("curvature_concave_frac"),
        intensity_hist=(np.average(np.array(hists), axis=0).tolist() if hists else []),
        n_images=int(sum(p.n_images for p in profiles)),
        source="pooled[" + " + ".join(p.source for p in profiles) + "]")


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


def _detect_ilm_rpe(img, smooth=7, axial_pitch_um=3.87, max_retina_um=900.0, border=10,
                    dc_skip=18):
    """Column-wise ILM (tissue onset) and RPE (deepest bright outer band), or (None, None).

    Two failure modes on these device recons are handled explicitly:

    * a bright **DC / zero-delay band** at the very top (the device crops it, our recon does not)
      would otherwise be picked as the ILM — the first ``dc_skip`` rows are ignored and the ILM is
      the first onset ABOVE a per-column vitreous baseline, not the first bright pixel;
    * the RPE is the **deepest** strong band, not the brightest thing in a shallow window — a search
      capped too shallow lands on the inner plexiform/outer-nuclear complex instead. The window now
      spans the full plausible retina (``max_retina_um``) and picks the deepest peak whose height is
      within ``rpe_rel`` of the column max, so the true outer complex is found even when an inner
      band is locally brighter.
    """
    H, W = img.shape
    top, bot = border + dc_skip, H - border
    if bot - top < 30:
        return None, None
    k = np.ones(smooth) / smooth
    col = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, img)
    max_th = int(max_retina_um / axial_pitch_um)
    rpe_rel = 0.55                                             # RPE peak must be >=55% of column max
    ilm = np.full(W, -1); rpe = np.full(W, -1)
    for c in range(W):
        v = col[top:bot, c]
        if v.size < 10:
            continue
        # vitreous baseline = low quantile of the (DC-skipped) column; ILM = onset above it
        base = float(np.percentile(v, 20)); cmax = float(v.max())
        if cmax - base < 1e-3:
            continue
        thr_ilm = base + 0.30 * (cmax - base)
        # sustained onset: the ILM begins a thick bright band, so require the next few rows to stay
        # bright too — rejects isolated DC/vitreous speckle spikes above the true tissue surface
        run = 6
        cs = np.cumsum(np.concatenate([[0.0], (v > thr_ilm).astype(float)]))
        sustained = np.where(cs[run:] - cs[:-run] >= run)[0]
        if sustained.size == 0:
            continue
        ilm[c] = sustained[0] + top
        lo = ilm[c] + 10
        hi = min(ilm[c] + max_th, bot)
        seg = col[lo:hi, c]
        if seg.size < 6:
            continue
        thr_rpe = rpe_rel * cmax
        strong = np.where(seg > thr_rpe)[0]                   # deepest strong band = RPE / Bruch's
        rpe[c] = lo + int(strong[-1] if strong.size else int(np.argmax(seg)))
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
    ax_grain, lat_grain, hists, contrast_db = [], [], [], []
    depth_env, rpe_curv = [], []
    n = 0
    for raw in images:
        img = _normalize(raw)
        if img is None:
            continue
        raw_arr = np.asarray(raw, float)              # for the un-normalised contrast (dB if <0)
        raw_lin = (10.0 ** (raw_arr / 20.0) if raw_arr.min() < -1.0
                   else np.clip(raw_arr, 1e-6, None))
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
        # 1b) un-normalised layer-mean contrast: dB range of the column-averaged (speckle-reduced)
        #     ILM->RPE profile — pins the layer brightness DYNAMIC RANGE (percentile norm hides it)
        prof_lin = np.zeros(N_TEMPLATE); mc = 0
        for c in range(0, W, 2):
            a, b = ilm[c], rpe[c]
            if b - a < 12:
                continue
            prof_lin += np.interp(a + axis * (b - a), np.arange(H), raw_lin[:, c]); mc += 1
        if mc:
            prof_lin /= mc
            ret = prof_lin[axis <= 1.0]               # ILM→RPE only (exclude the bright choroid margin,
            if ret.size:                              # else the choroid dilutes the layer contrast)
                pk = float(ret.max()); fl = max(float(ret.min()), pk * 1e-4)
                contrast_db.append(20.0 * np.log10(pk / fl))
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
        # FOV-normalized RPE sag (µm per (mm half-width)²): the bowl steepness comparable across FOVs.
        # SIGNED (keeps a2's sign) so D_curve penalises a dome-vs-bowl or over/under-curve mismatch, not
        # just magnitude; rows increase downward, so a2<0 = concave-up bowl (real posterior pole).
        fov_mm = W * lateral_pitch_um / 1000.0
        sag_um = a2 * (W / 2.0) ** 2 * axial_pitch_um
        rpe_curv.append(sag_um / ((fov_mm / 2.0) ** 2 + 1e-9))
        vit = img[:int(np.median(ilm)) - 5] if np.median(ilm) > 8 else img[:3]
        noise.append(float(np.median(vit)) if vit.size else 0.0)
        # 4) speckle grain via autocorrelation of a mid-retina patch
        ax_g, lat_g = _grain(img, ilm, rpe, axial_pitch_um, lateral_pitch_um)
        if ax_g:
            ax_grain.append(ax_g); lat_grain.append(lat_g)
        # 5) depth-intensity envelope: 8 slabs from the ILM down into the choroid, on LINEAR raw
        #    intensity, peak-normalized per image so the fall-off SHAPE (roll-off + deep attenuation)
        #    survives. Extends PAST the RPE (unlike the ILM→RPE template) to capture the deep fade.
        N_SLABS, slab_um = 8, 90.0
        env = np.zeros(N_SLABS); me = 0
        for c in range(0, W, 2):
            rows = ilm[c] + (np.arange(N_SLABS) + 0.5) * slab_um / axial_pitch_um
            if rows[-1] >= H:
                continue
            env += np.interp(rows, np.arange(H), raw_lin[:, c]); me += 1
        if me:
            env /= me
            depth_env.append((env / (env.max() + 1e-9)).tolist())
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
        retinal_contrast_db=float(np.median(contrast_db)) if contrast_db else 0.0,
        depth_intensity_env=(np.median(np.array(depth_env), axis=0).tolist() if depth_env else []),
        rpe_curvature_coef=float(np.median(rpe_curv)) if rpe_curv else 0.0,
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
