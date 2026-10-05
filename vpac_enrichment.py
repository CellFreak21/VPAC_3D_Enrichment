#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vpac_enrichment.py -- receptor enrichment in a subcellular compartment (3D confocal).

WHAT IT DOES
    For every cell in a field it builds the compartment mask from the marker channel
    (or from cell geometry, for CAAX), detects the nucleus, and reports one primary
    metric, the same for the four compartments:

        D = median(EGFP inside the mask) / median(EGFP in the reference)

        reference = cell - nucleus - inner dead band - mask volume

    The inner dead band is the shell within `dead_band_um` of the detected nuclear
    surface. It holds nucleoplasm the detector did not label and carries about half
    the EGFP of the rest of the cytoplasm, so it is excluded from the reference.

    D = 1 is the value expected with no effect, not the value measured with no
    effect. Each included cell therefore gets a Costes block-randomisation null:
    blocks of the receptor channel are shuffled inside mask + reference with the
    masks held fixed, and D is recomputed. A cell is called enriched above the
    97.5th percentile of that null and depleted below the 2.5th.

COMPARTMENTS
    caax     plasma membrane   geometric band inward from the cell surface
    fyve     endosomes         per-cell top-hat percentile, size and shape filtered
    golgi    Golgi             marker above a multiple of the per-cell median
    emerin   nuclear envelope  perinuclear ring at a fixed distance from the nucleus

INCLUSION GATES
    border, cell volume, receptor saturation spread, marker saturation, masked
    fraction, EGFP-positive fraction, a marker gate specific to each compartment,
    binucleate cells, and EGFP spatial spread (block coverage).

OUTPUT (one folder per run)
    results_<comp>_cells.csv    one row per cell
    results_<comp>_cells.xlsx   same table plus a legend sheet
    qc/                         per-field inclusion and mask overlays

USAGE
    python vpac_enrichment.py <folder|file.oib> [out] --comp caax|fyve|golgi|emerin
        --receptor VPAC1|VPAC2 [--use-labels] [--dz 0.35] [--no-null]
"""
import os, re, sys, glob
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import white_tophat, remove_small_objects
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import regionprops
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = os.environ.get('COLOC_BASE', os.path.expanduser('~/Colocalizacion'))
def resolve_out(p): return p if os.path.isabs(p) else os.path.join(BASE, p)

PREFIXES = {'caax': 'CAAX', 'fyve': 'FYVE', 'golgi': 'B3GALT', 'emerin': 'EMERIN'}

# Tag stored inside the label cache. The legacy spelling is kept on purpose: caches
# written by earlier versions must stay readable, so this is data, not code.
SEG_TAG = {'sum': 'suma', 'sum_norm': 'suma_norm', 'membrane': 'membrana'}
TIMES = ['120', '90', '60', '30', '15', '5']


# =========================== configuration ===========================
CFG = dict(
    ch_receptor=0, ch_marker=1,

    # --- segmentation (Cellpose, watershed fallback) ---
    use_cellpose=True, cellpose_batch=8, cellpose_mode='stitch',
    cp_flow=0.4, cp_prob=0.0, cp_stitch=0.25, cellpose_seg_channel='sum', cellpose_diam=60,
    seg_sigma_xy=2.0, seg_sigma_z=1.0, seg_otsu_frac=0.55, sep_min_um=12.0,
    only_egfp_positive=False, egfp_pos_factor=3.0,

    # --- inclusion gates ---
    min_vol_um3=400.0, max_vol_um3=6000.0,      # cell volume
    exclude_border=True,                        # touching the XY frame
    min_egfp_frac=0.13,                         # cytoplasm fraction above background+3sd
    saturation_value=4095,                      # detector ceiling
    max_sat_frac_largest=0.20,                  # saturated voxels in the LARGEST blob
    max_sat_frac_marker=0.30,
    mask_saturated=True, channels_to_mask=(0,), saturation_thr=4000, max_masked_frac=0.30,
    binuc_ratio_max=0.70,                       # 2nd dark body / 1st -> two nuclei
    cov_gate=True, cov_min=0.61,                # EGFP spatial spread
    cov_block_um=1.6, cov_min_frac=0.05, cov_min_vox_block=100,

    # --- nucleus detection ---
    exclude_nucleus=True,
    nuc_smooth_um=0.285,        # 3D closing+opening radius
    dark_pct=30,                # a voxel is dark below this per-cell percentile
    min_density=0.5, dark_win=7, min_nucleus_um3=15, nuc_center_weight=0.85,

    # --- reference ---
    dead_band_um=0.85,          # inner shell excluded from the reference

    # --- Costes null ---
    null_on=True, null_nperm=200, null_block=(2, 1, 1), null_seed=20260917,

    # --- masks ---
    membrane_band_um=0.70,                                   # caax
    tophat_radius_mask=8, fyve_mask_pct=97.0, mask_min_vox=10,  # fyve
    fyve_min_vol_um3=0.05, fyve_max_vol_um3=None, fyve_max_elong=5.0,
    golgi_mask_factor=4.0,                                   # golgi
    ring_inner_um=0.85, ring_outer_um=1.15,                  # emerin

    # --- marker gates ---
    caax_min_marker_frac=0.8, caax_max_concentration=10.0, caax_min_ratio=1.0,
    caax_marker_nsigma=3.0, caax_gate_band_um=0.83,   # band used by the gate only
    fyve_min_med_sigmas=15.0, fyve_max_gini=0.65,
    golgi_min_mask_vox=50,
    emerin_min_ratio=1.0, emerin_min_ring_vox=30,

    # --- QC figures ---
    qc_planes=5, qc_dpi=200, qc_scale=1.6,
)

# seg: channel given to Cellpose.  mask: how the compartment mask is built.
# nucleus_channel: channel used to find the dark nuclear body.
COMP = {
    'caax':   dict(seg='membrane', mask='membrane_band',   nucleus_channel='receptor'),
    'fyve':   dict(seg='sum',      mask='fyve_puncta',     nucleus_channel='receptor'),
    'golgi':  dict(seg='sum_norm', mask='golgi_blob',      nucleus_channel='receptor'),
    'emerin': dict(seg='sum_norm', mask='perinuclear_ring', nucleus_channel='sum'),
}


# =========================== io and segmentation ===========================
def read_stack(path, dz_forced=None):
    """Read an .oib stack. Returns (array [Z,C,Y,X], metadata)."""
    import tifffile
    meta = {}
    if path.lower().endswith('.oib'):
        import oiffile
        with oiffile.OifFile(path) as f:
            a = f.asarray().astype(np.float32); txt = str(f.mainfile)
        if a.ndim == 4 and a.shape[0] <= 5 and a.shape[1] > 5:
            a = np.transpose(a, (1, 0, 2, 3))
        elif a.ndim == 3:
            a = a[:, None, :, :]
        meta['raw'] = txt
        m = re.search(r'WidthConvertValue\s*=\s*([\d.]+)', txt)
        meta['px_um'] = round(float(m.group(1)), 4) if m else 0.207
        mz = re.search(r'Axis 3 Parameters Common:.*?Interval:\s*([\d.]+)', txt, re.S)
        meta['z_step_um'] = round(float(mz.group(1)) / 1000.0, 4) if mz else np.nan
    else:
        with tifffile.TiffFile(path) as tf:
            s = tf.series[0]; a = s.asarray().astype(np.float32); axes = s.axes
            ij = tf.imagej_metadata or {}
            meta['z_step_um'] = ij.get('spacing', np.nan)
            xr = tf.pages[0].tags.get('XResolution')
            meta['px_um'] = round(xr.value[1] / xr.value[0], 4) if xr else 0.207
            meta['raw'] = ij.get('Info', '') or ''
            if axes != 'ZCYX':
                order = [axes.index(c) for c in 'ZCYX' if c in axes]
                a = np.transpose(a, order)
    txt = meta.get('raw', '')
    def grab(pat, cast=float):
        mm = re.search(pat, txt)
        try: return cast(mm.group(1)) if mm else np.nan
        except: return np.nan

    hvs = re.findall(r'AnalogPMTVoltage\s*[:=]\s*(\d+)', txt)
    meta['hv_ch1'] = int(hvs[0]) if len(hvs) >= 1 else np.nan
    meta['hv_ch2'] = int(hvs[1]) if len(hvs) >= 2 else np.nan

    msq = re.search(r'SequentialMode:\s*(\S+)', txt)
    mode_ = msq.group(1) if msq else 'None'
    meta['sequential'] = mode_ not in ('None', 'none', 'NONE', None)
    meta['scan_mode'] = mode_
    meta['nZ'], meta['nC'], meta['nY'], meta['nX'] = a.shape
    if not np.isfinite(meta.get('px_um', np.nan)): meta['px_um'] = 0.207
    if dz_forced is not None:
        meta['z_step_um'] = dz_forced
    elif not np.isfinite(meta.get('z_step_um', np.nan)):
        print('   [AVISO] no se leyo el paso Z -> 0.4 por defecto (usa --dz)')
        meta['z_step_um'] = 0.4
    return a, meta


def _background_3d(a, cfg):
    """Per-image background level of one channel."""
    base = a[:, cfg['ch_receptor']] + a[:, cfg['ch_marker']]
    sm = ndi.gaussian_filter(base, (cfg['seg_sigma_z'], cfg['seg_sigma_xy'], cfg['seg_sigma_xy']))
    thr = threshold_otsu(sm) * cfg['seg_otsu_frac']
    fg = ndi.binary_fill_holes(ndi.binary_closing(sm > thr, np.ones((1, 5, 5))))
    return fg, None


def _keep_egfp_positive(a, lab, cfg):
    """Drop labels whose receptor signal is at background."""
    if not cfg.get('only_egfp_positive', False) or lab.max() == 0:
        return lab
    rec = a[:, cfg['ch_receptor']]; outside = lab == 0
    if outside.sum() < 100: return lab
    bg = np.median(rec[outside]); sd = rec[outside].std()
    thr = bg + cfg.get('egfp_pos_factor', 3.0) * sd
    labels = np.arange(1, lab.max() + 1)
    if cfg.get('egfp_metric', 'mean') == 'peak':
        vals = np.array([np.percentile(rec[lab == l], 99) if (lab == l).any() else 0 for l in labels])
    else:
        vals = np.asarray(ndi.mean(rec, labels=lab, index=labels))
    drop = labels[vals < thr]
    if len(drop):
        lab = lab.copy(); lab[np.isin(lab, drop)] = 0
    return lab


def labels_path(path, name, cfg):
    """Path of the cached label file for one image."""
    return os.path.join(os.path.dirname(path), name + '_labels.npz')


def load_labels(npz, cfg, name=''):
    """Load cached labels, rejecting them if built from another channel."""
    seg = SEG_TAG.get(cfg['cellpose_seg_channel'], cfg['cellpose_seg_channel'])
    try:
        z = np.load(npz)
    except Exception as e:
        return None, 'unreadable (%s)' % str(e)[:60]
    saved = str(z['seg']) if 'seg' in z.files else None
    if saved is None:
        print('   ! %s: cached labels carry no channel tag; assuming "%s".' % (name, seg))
        return z['lab'], 'legacy'
    if saved != seg:
        print('   ! %s: cached labels were built from "%s" but "%s" is needed.' % (name, saved, seg))
        print('     They are NOT overwritten. Delete the .npz yourself to re-segment.')
        return None, 'different channel'
    return z['lab'], 'ok'


def save_labels(npz, lab, px, dz, cfg):
    """Cache labels with the channel tag. Never overwrites an existing cache."""
    if os.path.exists(npz):
        print('   ! %s already exists; the new segmentation was NOT written.'
              % os.path.basename(npz))
        return
    np.savez_compressed(npz, lab=lab.astype(np.int32), px=px, dz=dz,
                        seg=SEG_TAG.get(cfg['cellpose_seg_channel'], cfg['cellpose_seg_channel']))


def _channel_sum_norm(a, cfg):
    """Both channels normalised and added, so neither one dominates."""
    rec, mar = a[:, cfg['ch_receptor']], a[:, cfg['ch_marker']]
    s = rec / (np.percentile(rec, 99) + 1) + mar / (np.percentile(mar, 99) + 1)
    return s.astype('float32')


def _channel_membrane(a, cfg):
    """Channel used to segment cells whose marker sits at the membrane."""
    rec, mar = a[:, cfg['ch_receptor']], a[:, cfg['ch_marker']]
    s = rec / (np.percentile(rec, 99) + 1) + mar / (np.percentile(mar, 99) + 1)
    th = np.stack([white_tophat(s[z], np.ones((5, 5))) for z in range(s.shape[0])])
    return (s + 2.0 * th).astype('float32')


def segment_3d(a, px, dz, cfg):
    """Segment cells in 3D with Cellpose, falling back to watershed."""
    if cfg['use_cellpose']:
        try:
            from cellpose import models
            import torch
            mdl = models.CellposeModel(gpu=torch.cuda.is_available())
            seg_src = cfg.get('cellpose_seg_channel', 'sum')
            if seg_src == 'membrane':     img = _channel_membrane(a, cfg)
            elif seg_src == 'sum_norm':  img = _channel_sum_norm(a, cfg)
            elif seg_src == 'receptor':   img = a[:, cfg['ch_receptor']].astype('float32')
            elif seg_src == 'marker':   img = a[:, cfg['ch_marker']].astype('float32')
            else:                         img = (a[:, cfg['ch_receptor']] + a[:, cfg['ch_marker']]).astype('float32')
            if cfg.get('cellpose_mode', 'stitch') == 'stitch':
                out = mdl.eval(img, z_axis=0, diameter=cfg['cellpose_diam'],
                               flow_threshold=cfg.get('cp_flow', 0.4),
                               cellprob_threshold=cfg.get('cp_prob', 0.0),
                               stitch_threshold=cfg.get('cp_stitch', 0.25),
                               batch_size=cfg.get('cellpose_batch', 8))
            else:
                out = mdl.eval(img, do_3D=True, z_axis=0, anisotropy=dz / px,
                               diameter=cfg['cellpose_diam'], flow_threshold=cfg.get('cp_flow', 0.4),
                               cellprob_threshold=cfg.get('cp_prob', 0.0), batch_size=cfg.get('cellpose_batch', 4))
            lab = out[0].astype(int)
            if lab.max() > 0:
                vox = px * px * dz
                for rp in regionprops(lab):
                    if rp.area * vox < cfg['min_vol_um3']:
                        lab[lab == rp.label] = 0
                return _keep_egfp_positive(a, lab, cfg), 'cellpose'
        except Exception as e:
            print('   (cellpose failed: %s -> watershed)' % str(e)[:100])
    fg, _ = _background_3d(a, cfg)
    from skimage.feature import peak_local_max
    dist = ndi.gaussian_filter(ndi.distance_transform_edt(fg, sampling=(dz, px, px)), (1, 2, 2))
    co = peak_local_max(dist, min_distance=int(round(cfg['sep_min_um'] / px)), labels=fg, exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    for i, c in enumerate(co, 1): markers[tuple(c)] = i
    if markers.max() == 0: markers, _ = ndi.label(fg)
    elev = ndi.gaussian_filter(a[:, cfg['ch_receptor']], (cfg['seg_sigma_z'], 2, 2))
    lab = _keep_egfp_positive(a, watershed(elev, markers, mask=fg), cfg)
    vox = px * px * dz
    for rp in regionprops(lab):
        if rp.area * vox < cfg['min_vol_um3']: lab[lab == rp.label] = 0
    return lab, 'watershed'



# =========================== per-cell measurements ===========================
def largest_saturated_fraction(rec, mask3d, cfg, nucleus=None):
    """Saturated fraction of the LARGEST saturated blob: spread, not amount."""

    if nucleus is not None:
        mask3d = mask3d & ~nucleus
    if mask3d.sum() == 0: return 0.0
    sat = mask3d & (rec >= cfg['saturation_value'])
    if not sat.any(): return 0.0
    axes_ = [np.where(np.any(sat, axis=tuple(j for j in range(3) if j != i)))[0] for i in range(3)]
    box = tuple(slice(int(v.min()), int(v.max()) + 1) for v in axes_)
    lo, n = ndi.label(sat[box], structure=np.ones((3, 3, 3)))
    if n == 0: return 0.0
    size = np.bincount(lo.ravel())[1:]
    return float(size.max()) / float(mask3d.sum())


def background_mode_sigma(mar, outside, cfg):
    """Background mode and left-flank sigma of the marker channel."""
    v = mar[outside]
    h, e = np.histogram(v, bins=np.arange(0, cfg['saturation_value'] + 5, 4))
    mode = float((e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2.0)
    left = v[v <= mode]
    sigma = float(max((mode - np.median(left)) / 0.6745, 1.0)) if left.size > 100 else 1.0
    return mode, sigma


def block_coverage(mk, rec, px, dz, cfg, nucleus=None):
    """Fraction of cytoplasmic blocks carrying EGFP: how spread the signal is."""

    if nucleus is None:
        return np.nan
    cyto = mk & ~nucleus
    if cyto.sum() < 200 or '_bg0m' not in cfg:
        return np.nan
    pos = cyto & (rec > cfg['_bg0m'] + 3 * cfg['_bg0s'])
    bz = max(1, int(round(cfg['cov_block_um'] / dz)))
    bxy = max(1, int(round(cfg['cov_block_um'] / px)))
    nz, ny, nx = cyto.shape
    iz = np.arange(nz) // bz; iy = np.arange(ny) // bxy; ix = np.arange(nx) // bxy
    bid = (iz[:, None, None] * (iy.max() + 1) + iy[None, :, None]) * (ix.max() + 1) \
        + ix[None, None, :]
    nb = int(bid.max()) + 1
    cnt_c = np.bincount(bid[cyto], minlength=nb)
    cnt_p = np.bincount(bid[pos], minlength=nb) if pos.any() else np.zeros(nb, int)
    ok = cnt_c >= cfg['cov_min_vox_block']
    if not ok.any():
        return np.nan
    fr = cnt_p[ok] / cnt_c[ok]
    return round(float((fr >= cfg['cov_min_frac']).mean()), 3)



# =========================== nucleus ===========================
def _ball_um(r_um, px, dz):
    """Spherical structuring element of a given physical radius."""
    rz = max(1, int(round(r_um / dz))); ry = max(1, int(round(r_um / px)))
    z, y, x = np.ogrid[-rz:rz + 1, -ry:ry + 1, -ry:ry + 1]
    return ((z * dz) ** 2 + (y * px) ** 2 + (x * px) ** 2) <= r_um ** 2 + 1e-9


def _largest_body_2d(plane, sem=None):
    """Largest connected dark body in one plane, optionally seeded."""
    lc, n = ndi.label(plane)
    if n == 0:
        return None
    ids = list(range(1, n + 1))
    if sem is not None:
        ids = [i for i in ids if (lc == i)[sem].any()]
        if not ids:
            return None
    size = {i: int((lc == i).sum()) for i in ids}
    return lc == max(size, key=size.get)


def _one_body_per_plane(nuc, mk, px, dz, r_smooth):
    """Keep one dark body per plane, chained across Z."""
    if nuc is None or not nuc.any():
        return nuc
    nz = nuc.shape[0]
    areas = []
    for z in range(nz):
        c = _largest_body_2d(nuc[z]) if nuc[z].any() else None
        areas.append(0 if c is None else int(c.sum()))
    z0 = int(np.argmax(areas))
    if areas[z0] == 0:
        return nuc
    out = np.zeros_like(nuc)
    out[z0] = _largest_body_2d(nuc[z0])
    for rng_ in (range(z0 - 1, -1, -1), range(z0 + 1, nz)):
        prev = out[z0].copy()
        for z in rng_:
            c = _largest_body_2d(nuc[z], prev) if nuc[z].any() else None
            if c is None:
                break
            out[z] = c
            prev = c
    for z in range(nz):
        if out[z].any():
            out[z] = ndi.binary_fill_holes(out[z])
    chain = out.copy()
    b = _ball_um(r_smooth, px, dz)
    out = ndi.binary_closing(out, b) & mk
    out = ndi.binary_opening(out, b)
    out = ndi.binary_fill_holes(out) & mk

    return out if out.any() else chain


def detect_nucleus(mk, signal, px, dz, cfg, return_info=False):
    """Nucleus = dark body of the cell. Returns the mask and the 2nd-body volume ratio."""

    dark_pct = cfg.get('dark_pct', 30); min_density = cfg.get('min_density', 0.5)
    win = cfg.get('dark_win', 7); min_nuc_vox = cfg.get('min_nucleus_um3', 15) / (px * px * dz)
    center_weight = cfg.get('nuc_center_weight', 0.85)
    vals = signal[mk]
    _empty = (None, dict(binuc_ratio=np.nan))
    if vals.size < 50:
        return _empty if return_info else None
    thr2 = np.percentile(vals, dark_pct)
    dark = (signal < thr2).astype(np.float32)
    dens = ndi.uniform_filter(dark, size=(1, win, win))
    nucleus = mk & (dens > min_density)
    nucleus = ndi.binary_closing(nucleus, np.ones((1, 3, 3)))
    nucleus = ndi.binary_opening(nucleus, np.ones((1, 2, 2))) & mk
    lc, n = ndi.label(nucleus)
    if n == 0:
        return _empty if return_info else None
    sizes = ndi.sum(np.ones_like(lc), lc, range(1, n + 1))
    cen = np.array(ndi.center_of_mass(mk)); rad = (mk.sum() / np.pi) ** 0.5 + 1e-6
    best, best_score = None, -1
    for i in range(1, n + 1):
        if sizes[i - 1] < min_nuc_vox:
            continue
        c = np.array(ndi.center_of_mass(lc == i)); d_center = np.linalg.norm((c - cen)[1:])
        center = max(0.0, 1.0 - d_center / (1.5 * rad))
        score = sizes[i - 1] * ((1 - center_weight) + center_weight * center)
        if score > best_score:
            best_score, best = score, i
    if best is None:
        return _empty if return_info else None

    large = sorted([sizes[i - 1] for i in range(1, n + 1) if sizes[i - 1] >= min_nuc_vox],
                     reverse=True)
    ratio = float(large[1] / large[0]) if len(large) > 1 else 0.0
    nuc = ndi.binary_fill_holes(lc == best)
    nuc = _one_body_per_plane(nuc, mk, px, dz, cfg.get('nuc_smooth_um', 0.285))
    return (nuc, dict(binuc_ratio=round(ratio, 3))) if return_info else nuc



# =========================== compartment masks ===========================
def fyve_mask(mk, marker, px, dz, cfg):
    """FYVE mask: per-cell top-hat percentile, filtered by size and shape."""
    r = cfg.get('tophat_radius_mask', 8)
    th = np.clip(np.stack([white_tophat(marker[z], np.ones((r, r))) for z in range(marker.shape[0])]), 0, None)
    T = np.percentile(th[mk], cfg.get('fyve_mask_pct', 97.0))
    lo, n = ndi.label(mk & (th >= T), structure=np.ones((3, 3, 3)))
    out = np.zeros(mk.shape, bool)
    if n == 0:
        return out
    vox = px * px * dz
    vmin = cfg.get('fyve_min_vol_um3', 0.05)
    vmax = cfg.get('fyve_max_vol_um3', None)
    emax = cfg.get('fyve_max_elong', 5.0)
    objs = ndi.find_objects(lo)
    for i in range(1, n + 1):
        sub = (lo[objs[i - 1]] == i)
        nv = int(sub.sum()); vol = nv * vox
        if vol < vmin: continue
        if vmax is not None and vol > vmax: continue
        if nv >= 8:
            z, y, x = np.nonzero(sub)
            P = np.stack([z * dz, y * px, x * px]).astype(float)
            P -= P.mean(1, keepdims=True)
            ev = np.clip(np.linalg.eigvalsh(np.cov(P) + 1e-9 * np.eye(3)), 1e-9, None)
            if np.sqrt(ev[2] / ev[0]) > emax: continue
        out[objs[i - 1]] |= sub
    return out


def golgi_mask(mk, marker, bg1m, bg1s, cfg):
    """Golgi mask: marker above a multiple of the per-cell median."""
    N = cfg.get('golgi_mask_factor', 4.0)
    med = np.median(marker[mk]) if mk.any() else 0.0
    T = N * max(med, bg1m + 2 * bg1s)
    m = mk & (marker > T)
    minv = cfg.get('mask_min_vox', 10)
    if minv and m.any():
        lo, n = ndi.label(m, structure=np.ones((3, 3, 3)))
        if n:
            sz = ndi.sum(np.ones_like(lo), lo, range(1, n + 1))
            keep = np.ones(n + 1, bool); keep[0] = False
            keep[np.where(sz < minv)[0] + 1] = False
            m = keep[lo]
    return m



# =========================== marker gates ===========================
def fyve_signal_quality(mk, marker, mode, sigma, cfg):
    """FYVE marker gate: signal above background and evenly spread."""
    v = marker[mk].astype(float)
    if v.size < 50:
        return False, dict(med_en_sigmas=np.nan, gini_marcador=np.nan)
    med_sig = float((np.median(v) - mode) / max(sigma, 1e-6))
    vb = np.sort(np.clip(v - mode, 0, None))
    s = vb.sum()
    n = len(vb)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(vb) / (n * s)) if s > 0 else np.nan
    m = dict(med_en_sigmas=round(med_sig, 1), gini_marcador=round(gini, 3) if np.isfinite(gini) else np.nan)
    ok = bool(med_sig >= cfg.get('fyve_min_med_sigmas', 15.0)
              and np.isfinite(gini) and gini <= cfg.get('fyve_max_gini', 0.65))
    return ok, m


def caax_marker_gate(mk, marker, px, mode, sigma, cfg, nucleus=None):
    """CAAX marker gate: own marker, spread out, and present at the membrane."""
    dxy = np.stack([ndi.distance_transform_edt(mk[z], sampling=(px, px)) for z in range(mk.shape[0])])
    band = mk & (dxy <= cfg.get('caax_gate_band_um', 0.83))
    interior = mk & ~band
    if nucleus is not None:
        interior = interior & ~nucleus
    m = dict(caax_pos_frac=np.nan, caax_concentration=np.nan, caax_ratio=np.nan,
             caax_median_int=np.nan, n_band=int(band.sum()), n_interior=int(interior.sum()))
    if band.sum() < 50 or interior.sum() < 50:
        return False, m
    vi = marker[interior]; vb = marker[band]
    T = mode + cfg.get('caax_marker_nsigma', 3.0) * sigma
    med_i = float(np.median(vi)); med_b = float(np.median(vb))
    m['caax_pos_frac'] = round(float((vi > T).mean()), 3)
    m['caax_median_int'] = round(med_i - mode, 1)
    m['caax_concentration'] = round(float((np.percentile(vi, 90) - mode) / max(med_i - mode, 1.0)), 1)
    m['caax_ratio'] = round(med_b / med_i, 3) if med_i > 0 else np.nan
    ok = bool(m['caax_pos_frac'] >= cfg.get('caax_min_marker_frac', 0.8)
              and m['caax_concentration'] <= cfg.get('caax_max_concentration', 10.0)
              and np.isfinite(m['caax_ratio'])
              and m['caax_ratio'] >= cfg.get('caax_min_ratio', 1.0))
    return ok, m



# =========================== naming and QC ===========================
def parse_condition(name, comp):
    """Read condition and field number from the file name."""
    pref = PREFIXES.get(comp, comp.upper())
    up = re.sub(r'^C\d+-', '', name.upper())
    if not up.startswith(pref): return 'other', ''
    rest = up[len(pref):]
    if rest.isdigit(): return 'basal', rest
    if rest.startswith('AV1'):   ag, rest = 'AV1', rest[3:]
    elif rest.startswith('AV2'): ag, rest = 'AV2', rest[3:]
    elif rest.startswith('V'):   ag, rest = 'VIP', rest[1:]
    else: return 'other', ''
    minutes = None
    for t in TIMES:
        if rest.startswith(t): minutes = int(t); rest = rest[len(t):]; break
    if minutes is None: return 'other', ''
    post = rest.startswith('P')
    if post: rest = rest[1:]
    if not rest.isdigit(): return 'other', ''
    return '%s%d%s' % (ag, minutes, 'P' if post else ''), rest


def _condition_rank(c):
    """Sort key for conditions: basal first, then by agonist and time."""
    if c == 'basal': return (0, 0, 0)
    m = re.match(r'(VIP|AV1|AV2)(\d+)(P?)$', str(c))
    if not m: return (9, 999, 0)
    return ({'VIP': 1, 'AV1': 2, 'AV2': 3}[m.group(1)], int(m.group(2)), 1 if m.group(3) else 0)


def save_qc(a, lab, status, path_png, cfg, cmasks=None, reasons=None):
    """QC figures: cell outlines coloured by inclusion, and mask overlays."""
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    rec, mar = a[:, cfg['ch_receptor']], a[:, cfg['ch_marker']]
    Z, H, W = rec.shape
    lo0, hi0 = np.percentile(rec, 1), np.percentile(rec, 99.5)
    lo1, hi1 = np.percentile(mar, 1), np.percentile(mar, 99.5)
    nz = lambda v, lo, hi: np.clip((v - lo) / max(hi - lo, 1), 0, 1)
    n_pan = min(cfg.get('qc_planes', 5), Z)
    zs = np.linspace(0, Z - 1, n_pan).round().astype(int)
    dpi = cfg.get('qc_dpi', 200)
    inches = W / dpi * cfg.get('qc_scale', 1.6)

    COL = {'included': [0, 1, 0], 'saturated': [1, 0, 1], 'marker_negative': [1, 0.25, 0.25],
           'egfp_negative': [1, 0.6, 0], 'other': [0.55, 0.55, 0.55]}
    def verdict(l):
        if status.get(l) == 'included':
            return 'included'
        m = str((reasons or {}).get(l, ''))
        if 'saturated' in m or 'masked' in m: return 'saturated'
        if 'marker_negative' in m: return 'marker_negative'
        if 'egfp_negative' in m: return 'egfp_negative'
        return 'other'

    fig, axes = plt.subplots(1, len(zs), figsize=(inches * len(zs), inches * 1.06))
    if len(zs) == 1: axes = [axes]
    for ax, z in zip(axes, zs):
        rgb = np.zeros((H, W, 3))
        rgb[..., 1] = nz(rec[z], lo0, hi0); rgb[..., 0] = nz(mar[z], lo1, hi1)
        rgb[(rec[z] >= cfg['saturation_value']) | (mar[z] >= cfg['saturation_value'])] = [0.2, 0.4, 1.0]
        for rp in regionprops(lab[z]):
            rgb[find_boundaries(lab[z] == rp.label, mode='outer')] = COL[verdict(rp.label)]
        ax.imshow(rgb, interpolation='nearest')
        for rp in regionprops(lab[z]):
            yy, xx = rp.centroid
            ax.text(xx, yy, str(rp.label), color='yellow', fontsize=5.5, weight='bold',
                    ha='center', va='center')
        ax.set_title('z=%d  (Fiji %d)' % (z, z + 1), fontsize=9); ax.axis('off')
    counts = {k: sum(1 for l in status if verdict(l) == k) for k in COL}
    fig.suptitle('green = included (%d)   magenta = saturated/masked (%d)   '
                 'red = marker-negative (%d)   orange = EGFP-negative (%d)   '
                 'gray = other (%d)   |   blue = saturated voxel'
                 % (counts['included'], counts['saturated'], counts['marker_negative'],
                    counts['egfp_negative'], counts['other']), fontsize=10)
    fig.tight_layout()
    fig.savefig(path_png.replace('.png', '_inclusion.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if cmasks:

        m_in = np.zeros(lab.shape, bool); m_out = np.zeros(lab.shape, bool)
        for l, cm in cmasks.items():
            if cm is None or cm.shape != lab.shape:
                continue
            if status.get(l) == 'included':
                m_in |= cm
            else:
                m_out |= cm
        vmax = float(np.percentile(mar[lab > 0], 99.5)) if (lab > 0).any() else float(mar.max())
        fig, axes = plt.subplots(1, len(zs), figsize=(inches * len(zs), inches * 1.06))
        if len(zs) == 1: axes = [axes]
        for ax, z in zip(axes, zs):
            im = ax.imshow(mar[z], cmap='inferno', vmin=0, vmax=vmax, interpolation='nearest')
            if m_out[z].any():
                ax.contour(m_out[z], levels=[0.5], colors='#888888', linewidths=0.4,
                           linestyles='dotted')
            if m_in[z].any():
                ax.contour(m_in[z], levels=[0.5], colors='cyan', linewidths=0.6)
            ax.set_title('z=%d  (Fiji %d)' % (z, z + 1), fontsize=9); ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle('MARKER (inferno, true scale 0-%.0f)  |  cyan = mask of an INCLUDED cell   '
                     'dotted gray = mask of an EXCLUDED cell' % vmax,
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(path_png.replace('.png', '_mascaras.png'), dpi=dpi, bbox_inches='tight')
        plt.close(fig)



# =========================== reference, metric and null ===========================
def perinuclear_ring(mk, nucleus, dn, cfg):
    """Emerin mask: shell between `ring_inner_um` and `ring_outer_um` from the nucleus."""
    if nucleus is None or dn is None:
        return np.zeros_like(mk)
    return mk & ~nucleus & (dn >= cfg['ring_inner_um']) & (dn <= cfg['ring_outer_um'])


def membrane_band(mk, nucleus, px, dz, cfg):
    """CAAX mask: geometric band inward from the cell surface."""
    d_edge = ndi.distance_transform_edt(mk, sampling=(dz, px, px))
    band = mk & (d_edge <= cfg['membrane_band_um'])
    return band & ~nucleus if nucleus is not None else band


def reference_mask(mk, comp_mask, nucleus, dn, cfg):
    """Reference volume: cell minus nucleus, minus the inner dead band, minus the mask."""
    ref = mk & ~comp_mask
    if nucleus is not None:
        ref = ref & ~nucleus
    band = cfg.get('dead_band_um')
    if dn is not None and band:
        ref = ref & (dn > band)
    return ref


def enrichment_D(recbg, comp_mask, ref, min_vox=50):
    """D = median(EGFP in mask) / median(EGFP in reference). Returns (D, num, den)."""
    if comp_mask.sum() < min_vox or ref.sum() < min_vox:
        return np.nan, np.nan, np.nan
    num = float(np.median(recbg[comp_mask])); den = float(np.median(recbg[ref]))
    if not (np.isfinite(den) and den > 0):
        return np.nan, num, den
    return float(num / den), num, den


def costes_null(recbg, comp_mask, ref, block=(2, 1, 1), nperm=200, seed=20260917):
    """Null distribution of D from block randomisation of the receptor, masks fixed.

    Only blocks at least half inside mask+reference are shuffled. `mask_fraction`
    reports how much of the mask ended up inside shuffled blocks; below ~1.0 the
    null is contaminated and must not be used."""
    if comp_mask.sum() < 50 or ref.sum() < 50:
        return None
    R = np.ascontiguousarray(recbg, dtype=np.float32)
    universe = comp_mask | ref
    bz, by, bx = block
    sh = R.shape
    nz, ny, nx = -(-sh[0] // bz), -(-sh[1] // by), -(-sh[2] // bx)
    pad = ((0, nz * bz - sh[0]), (0, ny * by - sh[1]), (0, nx * bx - sh[2]))

    def to_blocks(A):
        return np.pad(A, pad).reshape(nz, bz, ny, by, nx, bx) \
                 .transpose(0, 2, 4, 1, 3, 5).reshape(nz * ny * nx, bz * by * bx)

    RB = to_blocks(R); UB = to_blocks(universe)
    idx = np.where(UB.mean(axis=1) >= 0.5)[0]
    if idx.size < 20:
        return None
    MB = to_blocks(comp_mask)
    frac = float(MB[idx].sum() / max(1, MB.sum()))
    rng = np.random.default_rng(seed)
    base = RB.copy(); out = []
    for _ in range(int(nperm)):
        RB2 = base.copy()
        RB2[idx] = base[rng.permutation(idx)]
        Rn = RB2.reshape(nz, ny, nx, bz, by, bx).transpose(0, 3, 1, 4, 2, 5) \
                .reshape(nz * bz, ny * by, nx * bx)[:sh[0], :sh[1], :sh[2]]
        d = np.median(Rn[ref])
        out.append(np.median(Rn[comp_mask]) / d if d > 0 else np.nan)
    v = np.asarray(out, float); v = v[np.isfinite(v)]
    if v.size < 10:
        return None
    return dict(null_p2_5=round(float(np.percentile(v, 2.5)), 4),
                null_p50=round(float(np.percentile(v, 50)), 4),
                null_p97_5=round(float(np.percentile(v, 97.5)), 4),
                null_mask_fraction=round(frac, 3))


def null_verdict(d, lo, hi):
    """Call a cell enriched, depleted or ns against its own null."""
    if not (np.isfinite(d) and np.isfinite(lo) and np.isfinite(hi)):
        return ''
    return 'enriched' if d > hi else ('depleted' if d < lo else 'ns')


# =========================== output ===========================
COLUMNS = ['receptor', 'image', 'condition', 'field', 'cell', 'included', 'reason',
           'cell_volume_um3', 'mask_volume_um3', 'mask_median', 'reference_median',
           'enrichment_D', 'block_coverage', 'nucleus_fraction',
           'null_p2_5', 'null_p50', 'null_p97_5', 'null_mask_fraction', 'null_verdict']

LEGEND = [
    ('receptor', 'Receptor construct (VPAC1 / VPAC2).'),
    ('image', 'Source file name, without extension.'),
    ('condition', 'Treatment read from the file name: basal, VIP<min>, AV1<min>, AV2<min>.'),
    ('field', 'Field number within that condition.'),
    ('cell', 'Label of the cell in the segmentation.'),
    ('included', 'TRUE if the cell passes every gate and enters the analysis.'),
    ('reason', 'Gate(s) failed: border, too_large, too_small, saturated, masked, '
               'egfp_negative, marker_negative, binucleate, egfp_spread.'),
    ('cell_volume_um3', 'Segmented cell volume.'),
    ('mask_volume_um3', 'Compartment mask volume.'),
    ('mask_median', 'Median background-subtracted EGFP inside the mask.'),
    ('reference_median', 'Median background-subtracted EGFP in the reference: '
                         'cell minus nucleus, minus the inner dead band, minus the mask.'),
    ('enrichment_D', 'PRIMARY METRIC. mask_median / reference_median.'),
    ('block_coverage', 'Fraction of cytoplasmic blocks carrying EGFP. Low means the '
                       'signal sits in part of the object only.'),
    ('nucleus_fraction', 'Detected nucleus volume over cell volume.'),
    ('null_p2_5', 'Costes null, 2.5th percentile. Below it the mask is depleted.'),
    ('null_p50', 'Costes null, median. Should be about 1.000; if not, the block size is wrong.'),
    ('null_p97_5', 'Costes null, 97.5th percentile. Above it the mask is enriched.'),
    ('null_mask_fraction', 'Fraction of the mask inside shuffled blocks. Below ~1.0 the '
                           'null is contaminated and the verdict must not be used.'),
    ('null_verdict', 'enriched / depleted / ns, from enrichment_D against this cell own null.'),
]


def sort_table(df):
    """Order rows by condition, then field, then included first, then cell."""
    df = df.copy()
    df['_c'] = df.condition.map(_condition_rank)
    df = df.sort_values(['_c', 'condition', 'image', 'included', 'cell'],
                        ascending=[True, True, True, False, True])
    return df.drop(columns='_c')


def write_xlsx(df, path):
    """One sheet with the table and one sheet with the legend."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook(); ws = wb.active; ws.title = 'cells'
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = PatternFill('solid', fgColor='DDEBF7')
    grey = PatternFill('solid', fgColor='F2F2F2')
    for _, r in df.iterrows():
        ws.append([None if (isinstance(r[c], float) and not np.isfinite(r[c])) else r[c]
                   for c in COLUMNS])
        if not bool(r['included']):
            for c in ws[ws.max_row]:
                c.fill = grey
    ws.freeze_panes = 'A2'
    for i, c in enumerate(COLUMNS, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = max(11, min(22, len(c) + 3))
    wl = wb.create_sheet('legend')
    wl.append(['column', 'meaning'])
    for c in wl[1]:
        c.font = Font(bold=True); c.fill = PatternFill('solid', fgColor='DDEBF7')
    for k, v in LEGEND:
        wl.append([k, v])
    wl.column_dimensions['A'].width = 22; wl.column_dimensions['B'].width = 110
    for row in wl.iter_rows(min_row=2):
        row[1].alignment = row[1].alignment.copy(wrap_text=True)
    wb.save(path)


# =========================== main loop ===========================
def process(path, comp, cfg, out_dir, use_labels, dz_forced, receptor):
    """Measure every cell of one field and return its rows."""
    name = os.path.splitext(os.path.basename(path))[0]
    cond, field = parse_condition(name, comp)
    a, meta = read_stack(path, dz_forced)
    px, dz = meta['px_um'], meta['z_step_um']
    rec, mar = a[:, cfg['ch_receptor']], a[:, cfg['ch_marker']]

    npz = labels_path(path, name, cfg)
    lab = None
    if use_labels and os.path.exists(npz):
        lab, _ = load_labels(npz, cfg, name)
    if lab is None:
        lab, _ = segment_3d(a, px, dz, cfg); save_labels(npz, lab, px, dz, cfg)

    outside = lab == 0
    bg0m, bg0s = float(np.median(rec[outside])), float(rec[outside].std())
    bg1m, bg1s = float(np.median(mar[outside])), float(mar[outside].std())
    cfg['_bg0m'], cfg['_bg0s'] = bg0m, bg0s
    recbg_full = np.clip(rec - bg0m, 0, None)
    mode_mar, sigma_mar = background_mode_sigma(mar, outside, cfg)

    signal_sum = ndi.gaussian_filter((rec + mar).astype(np.float32), (1, 1.5, 1.5))
    signal_nuc = signal_sum if COMP[comp]['nucleus_channel'] == 'sum' else \
        ndi.gaussian_filter(rec.astype(np.float32), (1, 1.5, 1.5))

    sl = ndi.find_objects(lab)
    border_ids = set(np.unique(np.concatenate([lab[:, 0, :].ravel(), lab[:, -1, :].ravel(),
                    lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))) - {0}
    kind = COMP[comp]['mask']
    rows, status, cmasks, reasons_qc = [], {}, {}, {}

    for rp in regionprops(lab):
        l = rp.label; s = sl[l - 1]
        if s is None:
            continue
        sz = (slice(max(0, s[0].start - 2), min(lab.shape[0], s[0].stop + 2)),
              slice(max(0, s[1].start - 3), min(lab.shape[1], s[1].stop + 3)),
              slice(max(0, s[2].start - 3), min(lab.shape[2], s[2].stop + 3)))
        mk = lab[sz] == l
        recc, marker = rec[sz], mar[sz]
        recbg = np.ascontiguousarray(recbg_full[sz])
        vol = rp.area * px * px * dz

        # nucleus and distance to its surface
        nucleus, binuc = None, np.nan
        if cfg.get('exclude_nucleus', True):
            nucleus, info = detect_nucleus(mk, signal_nuc[sz], px, dz, cfg, return_info=True)
            binuc = info['binuc_ratio']
        dn = ndi.distance_transform_edt(~nucleus, sampling=(dz, px, px)) \
            if (nucleus is not None and nucleus.any()) else None
        cyto = mk & ~nucleus if nucleus is not None else mk

        # compartment mask
        if kind == 'membrane_band':
            cmask = membrane_band(mk, nucleus, px, dz, cfg)
        elif kind == 'fyve_puncta':
            cmask = fyve_mask(mk, marker, px, dz, cfg)
        elif kind == 'golgi_blob':
            cmask = golgi_mask(mk, marker, bg1m, bg1s, cfg)
        else:
            cmask = perinuclear_ring(mk, nucleus, dn, cfg)
        ref = reference_mask(mk, cmask, nucleus, dn, cfg)

        # saturation and masked fraction
        sat_marker = float((marker >= cfg['saturation_value'])[mk].mean())
        masked_frac = 0.0
        if cfg['mask_saturated']:
            sat_vox = np.zeros(mk.shape, bool)
            if cfg['ch_receptor'] in cfg['channels_to_mask']:
                sat_vox |= (recc >= cfg['saturation_thr'])
            if cfg['ch_marker'] in cfg['channels_to_mask']:
                sat_vox |= (marker >= cfg['saturation_thr'])
            n_cell = int(mk.sum())
            masked_frac = 1.0 - (mk & ~sat_vox).sum() / n_cell if n_cell else 1.0

        # EGFP amount and spread, both over the cytoplasm
        egfp_frac = float((recc[cyto] > bg0m + 3 * bg0s).mean()) if cyto.sum() >= 50 else np.nan
        sat_largest = largest_saturated_fraction(recc, mk, cfg, nucleus)
        coverage = block_coverage(mk, recc, px, dz, cfg, nucleus)

        # marker gate
        if comp == 'caax':
            marker_ok, _ = caax_marker_gate(mk, marker, px, mode_mar, sigma_mar, cfg, nucleus)
        elif comp == 'fyve':
            marker_ok, _ = fyve_signal_quality(mk, marker, mode_mar, sigma_mar, cfg)
        elif comp == 'golgi':
            marker_ok = cmask.sum() >= cfg['golgi_min_mask_vox']
        else:
            rest_e = mk & ~cmask
            if nucleus is not None:
                rest_e = rest_e & ~nucleus
            ratio_e = float(marker[cmask].mean() / marker[rest_e].mean()) \
                if (cmask.sum() >= cfg['emerin_min_ring_vox'] and rest_e.sum() >= 30
                    and marker[rest_e].mean() > 0) else np.nan
            marker_ok = bool(np.isfinite(ratio_e) and ratio_e >= cfg['emerin_min_ratio'])

        reasons = []
        if cfg['exclude_border'] and l in border_ids: reasons.append('border')
        if vol > cfg['max_vol_um3']: reasons.append('too_large')
        if vol < cfg['min_vol_um3']: reasons.append('too_small')
        if sat_largest > cfg['max_sat_frac_largest'] or sat_marker > cfg['max_sat_frac_marker']:
            reasons.append('saturated')
        if masked_frac > cfg['max_masked_frac']: reasons.append('masked')
        if not (np.isfinite(egfp_frac) and egfp_frac >= cfg['min_egfp_frac']):
            reasons.append('egfp_negative')
        if not marker_ok: reasons.append('marker_negative')
        if np.isfinite(binuc) and binuc >= cfg['binuc_ratio_max']: reasons.append('binucleate')
        if cfg['cov_gate'] and np.isfinite(coverage) and coverage < cfg['cov_min']:
            reasons.append('egfp_spread')
        included = len(reasons) == 0

        d, num, den = enrichment_D(recbg, cmask, ref)
        row = dict(receptor=receptor, image=name, condition=cond, field=field, cell=l,
                   included=included, reason=';'.join(reasons),
                   cell_volume_um3=round(vol, 1),
                   mask_volume_um3=round(float(cmask.sum() * px * px * dz), 3),
                   mask_median=round(num, 1) if np.isfinite(num) else np.nan,
                   reference_median=round(den, 1) if np.isfinite(den) else np.nan,
                   enrichment_D=round(d, 4) if np.isfinite(d) else np.nan,
                   block_coverage=coverage,
                   nucleus_fraction=round(float(nucleus.sum() / mk.sum()), 4)
                   if nucleus is not None else np.nan,
                   null_p2_5=np.nan, null_p50=np.nan, null_p97_5=np.nan,
                   null_mask_fraction=np.nan, null_verdict='')

        # the null is expensive, so only included cells get one
        if included and cfg.get('null_on', True) and cmask.any():
            nul = costes_null(recbg, cmask, ref, cfg['null_block'], cfg['null_nperm'],
                              cfg['null_seed'])
            if nul:
                row.update(nul)
                row['null_verdict'] = null_verdict(d, nul['null_p2_5'], nul['null_p97_5'])

        rows.append(row)
        status[l] = 'included' if included else reasons[0]
        reasons_qc[l] = row['reason']
        full = np.zeros(lab.shape, bool); full[sz] = cmask; cmasks[l] = full

    save_qc(a, lab, status, os.path.join(out_dir, 'qc', name + '.png'), cfg, cmasks, reasons_qc)
    n_in = sum(1 for r in rows if r['included'])
    print('   %-14s %-8s %3d cells, %3d included' % (name, cond, len(rows), n_in), flush=True)
    return rows


def main():
    ap = sys.argv[1:]
    if not ap:
        print(__doc__); sys.exit(0)
    src = ap[0]
    out = ap[1] if len(ap) > 1 and not ap[1].startswith('--') else 'out'
    def opt(flag, default=None):
        return ap[ap.index(flag) + 1] if flag in ap else default
    comp = (opt('--comp', 'caax') or 'caax').lower()
    if comp == 'emerina':
        comp = 'emerin'
    if comp not in COMP:
        sys.exit('unknown compartment: %s (use %s)' % (comp, '/'.join(COMP)))
    receptor = opt('--receptor', '')
    dz_forced = float(opt('--dz')) if opt('--dz') else None
    use_labels = '--use-labels' in ap or '--usar_labels' in ap

    cfg = dict(CFG); cfg.update(COMP[comp])
    cfg['cellpose_seg_channel'] = COMP[comp]['seg']
    if '--no-null' in ap:
        cfg['null_on'] = False

    files = sorted(glob.glob(os.path.join(src, '*.oib'))) if os.path.isdir(src) else [src]
    out_dir = resolve_out(out); os.makedirs(out_dir, exist_ok=True)
    print('comp=%s receptor=%s | %d file(s) | out=%s' % (comp, receptor, len(files), out_dir))

    rows = []
    for f in files:
        rows += process(f, comp, cfg, out_dir, use_labels, dz_forced, receptor)
    df = sort_table(pd.DataFrame(rows)[COLUMNS])
    csv = os.path.join(out_dir, 'results_%s_cells.csv' % comp)
    df.to_csv(csv, index=False)
    write_xlsx(df, os.path.join(out_dir, 'results_%s_cells.xlsx' % comp))

    ok = df[df.included]
    print('\n== %d cells, %d included ==' % (len(df), len(ok)))
    if len(ok):
        g = ok.groupby('condition').agg(n=('cell', 'size'),
                                        D_median=('enrichment_D', 'median'),
                                        enriched=('null_verdict', lambda x: (x == 'enriched').sum()),
                                        depleted=('null_verdict', lambda x: (x == 'depleted').sum()))
        g['_o'] = [ _condition_rank(c) for c in g.index ]
        print(g.sort_values('_o').drop(columns='_o').round(4).to_string())
    print('\nOutput in:', out_dir)


if __name__ == '__main__':
    main()
