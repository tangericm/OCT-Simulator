# Non-Rayleigh speckle in retinal OCT — findings & the octsim model

Why iterative calibration plateaued at pool-mean D ≈ 3.40, and the physics-grounded fix.

## The plateau diagnosis (2026-07-20)

After the shared-optics loop converged (two consecutive DE runs failing to improve >2%), the
per-corpus residual was dominated by **D_hist** — the tissue intensity-histogram distance — not the
layer-brightness template (D_template ≈ 0.04) or speckle contrast (≈0.37, matched):

| corpus | D_total | D_hist | D_template | D_speckle |
|---|---|---|---|---|
| m3_macula_6mm | 10.43 | **8.83** | 0.04 | 1.56 |
| octc8 | 3.78 | **2.55** | 0.02 | 0.35 |
| octid | 1.99 | **1.35** | 0.04 | 0.60 |

The forward model drew a **single fixed-variance circular-Gaussian** reflectivity per cell →
fully-developed **Rayleigh** amplitude everywhere (intensity contrast 1). No θ parameter changed the
speckle *statistics*, so the optimizer had no lever on the histogram shape. That is a forward-model
gap, not a tuning failure.

## What the literature says (deep-research, 16 sources, 22/25 claims adversarially confirmed)

1. **Rayleigh is the many-scatterer limit.** OCT amplitude is Rayleigh (circular-Gaussian, CLT) only
   with ≳10 randomly-placed scatterers per resolution cell; amplitude contrast → √(4/π−1) ≈ **0.52**.
   With few scatterers the statistics go **non-Rayleigh (K-distribution)**, heavier-tailed, contrast
   rising above 0.52. [Almasian *Sci Rep* 2017; Niemczyk & Iskander 2021; review PMC8919025]
2. **The shape parameter is a (confounded) scatterer-density proxy**, α→∞ recovering Rayleigh; it also
   absorbs clustering, cross-section variance, coherent-to-diffuse ratio — not density alone.
   [Niemczyk & Iskander 2021; Jesus & Iskander *BOE* 2016]
3. **Retina-specific per-layer values** (the one directly-usable table): raw *linear* intensity fits a
   stretched-exponential p(I)=k·exp(−(I/λ)^β). Per layer: **IS/RPE β≈0.69 (most non-Rayleigh)**,
   NFL β≈0.81, GCL/IPL/INL/OPL β≈0.78, **ONL β≈1.07 (near-Rayleigh)**. [Grzywacz *et al.*, IEEE TMI
   29(6):1224, 2010]. Note the ordering is **not** naive density (RPE is dense yet most non-Rayleigh)
   — so a per-layer M must follow measured β, not scatterer count.
4. **Real tissue is super-Rayleigh**: intensity contrast 1.28–2.00; power-law Burr/Lomax (scale d,
   exponent b≈3–6) beats Rayleigh in 8/9 tissues. [Ge, Rolland & Parker *BOE* 2021]
5. **Minimal correct extension**: a **compound / scale-mixed Rayleigh** — draw the per-cell variance
   from a gamma (→K) or power-law (→Lomax) mixing distribution; keep a scale term to hold brightness,
   one shape term for the tail. A single Rayleigh draw *structurally cannot* make these tails.
   [synthesis of Kirillin *Opt Lett* 2014 + Parker 2021]
6. **Coupling caveat (load-bearing):** for these families the mean depends on *both* scale and shape,
   so changing the tail shifts brightness — the scale must be re-fit (or the modulator mean pinned) to
   hold the layer-brightness template. Brightness and histogram-shape are *coupled*, not independent.
7. Refuted / excluded: sparse contrast∝concentration as a brightness-preserving knob (0-3);
   gamma/log-normal as a universal per-tissue model (0-3); mixture-of-exponentials as the retinal tail
   mechanism (1-2). octsim relies on none of these.

## The octsim model (v1 shipped)

Compound-K by Gamma-modulating the circular-Gaussian variance (`sample.py`,
`build_reflectivity_from_boundaries`):

```
g ~ CN(0,1)                          # unit circular-Gaussian (|g| Rayleigh, E|g|²=1)
tex ~ Gamma(M, 1/M)                  # E[tex]=1, intensity var 1/M ; M = eff. #scatterers/cell
tex = correlate_to_resolution_cell(tex), restore var→1/M, renormalize E[tex]=1
amplitude = g · sqrt(tex)            # intensity = |g|²·tex  ⇒  E[intensity] UNCHANGED
```

- **E[tex]=1 exactly** → mean reflectivity, layer-brightness template, interface reflectors, and
  Beer–Lambert roll-off all preserved (directly answers caveat #6). Intensity contrast → √(1+2/M);
  M→∞ recovers today's Rayleigh look.
- **Correlated to several resolution cells** (`_K_AXIAL_GRAIN_PX`, `_K_LATERAL_GRAIN_BEAMS`) with the
  marginal variance *restored* after smoothing, so the modulator survives the reconstruction PSF /
  beam filter as a slow envelope (factors out: intensity ≈ tex·Rayleigh). An iid-per-cell modulator is
  averaged back to Rayleigh (CLT) and is inert — this grain-matching is what makes M identifiable.
- **Label map bit-identical**: labels come from geometry upstream of the speckle draw; the modulator
  uses a spawned RNG child so downstream features are unchanged too. (Verified + gated by
  `tests/test_calibrate.py::test_optics_never_touch_labels`.)
- **Single global scalar** M (log-fit, M∈[2,50], `None`=Rayleigh sentinel at the top). Per-layer M is
  deliberately **not** fitted: `extract_texture` aggregates all ILM→RPE pixels into ONE histogram, so
  a per-layer vector is under-determined. A fixed per-layer prior mapped from Grzywacz β is the v2
  option (kept frozen, never added to θ).

Identifiability + recovery proven by `tests/test_calibrate.py::test_speckle_shape_recovery`
(unique D minimum at the injected M*, DE recovers it, speckle contrast stays in-gate).

## Measured lever (8-seed sweep, literature optics)

| M | speckle contrast | hist χ² vs Rayleigh |
|---|---|---|
| None (Rayleigh) | 0.325 | 0.000 |
| 25 | 0.327 | 0.007 |
| 8 | 0.332 | 0.031 |
| 4 | 0.337 | 0.131 |
| 2 | 0.353 | 0.519 |

Monotone and identifiable; contrast rises modestly (super-Rayleigh direction), staying near the real
M3 frame-averaged 0.34–0.38 — so a low M can help D_speckle as well as D_hist.

## Open questions (from the research; future work)
- Retina-*layer*-specific K/Burr parameters (only stretched-exp β/λ and one GGD-GCL scale are
  published; K/Burr known only for phantoms/cornea/brain).
- Quantitative map from raw-linear tissue statistics → clinical 8-bit log/tone-mapped/averaged
  histogram (only the qualitative log-suppression effect is documented — Kobayashi *JOSA A* 2011).
  Relevant to octc8/octid, whose targets are 8-bit clinical JPEGs (real speckle ≈0.46).

### Primary sources
- Almasian et al., *OCT Amplitude and Speckle Statistics of Discrete Random Media*, Sci Rep 2017 — nature.com/articles/s41598-017-14115-3
- Grzywacz et al., *Statistics of OCT Data From Human Retina*, IEEE TMI 29(6):1224, 2010 — pmc.ncbi.nlm.nih.gov/articles/PMC2922066/
- Ge, Rolland & Parker, *Speckle statistics of biological tissues in OCT*, BOE 12(7):4179, 2021 — pmc.ncbi.nlm.nih.gov/articles/PMC8367221/
- Jesus & Iskander, *Generalized-gamma corneal speckle*, BOE 2016 — pmc.ncbi.nlm.nih.gov/articles/PMC5231290/
- Niemczyk & Iskander 2021 — arxiv.org/pdf/2107.07256 ; Kobayashi et al., JOSA A 28(4):517, 2011 — pubmed.ncbi.nlm.nih.gov/21478944/
- Review: *Signal-carrying speckle in OCT* — pmc.ncbi.nlm.nih.gov/articles/PMC8919025/
