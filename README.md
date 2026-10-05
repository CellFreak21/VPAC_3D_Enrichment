# vpac-3d-enrichment

Quantification of receptor enrichment in subcellular compartments from 3D confocal stacks.

This repository contains the image-analysis pipeline used in the doctoral thesis
"VPAC1 and VPAC2 receptor signalling and trafficking. Functional implications for human regulatory T cells" (Alicia Cabrera Martín, Universidad Complutense de Madrid, 2026) to measure where the VIP receptors
VPAC1 and VPAC2 (EGFP fusions) accumulate in four compartments, each labelled with an
mRuby3 marker:

| Compartment | Marker | How the mask is built |
|---|---|---|
| Plasma membrane | mRuby3-CAAX | Geometric band, 0.70 µm inward from the cell surface |
| Early endosomes | mRuby3-FYVE | Per-cell top-hat, 97th percentile, size and shape filtered |
| Golgi apparatus | B4GALT1-mRuby3 | Marker above 4 × the per-cell median |
| Nuclear envelope | mRuby3-emerin | Shell between 0.85 and 1.15 µm from the nuclear surface |

Every mask is defined from the marker channel (or from cell geometry), never from the
receptor, so the region where the receptor is measured does not depend on the receptor.

## What the pipeline does

For each `.oib` stack (Olympus FluoView):

1. **Cell segmentation** in 3D with Cellpose (pretrained model, no retraining), using
   both channels. Labels are cached next to the image (`<name>_labels.npz`).
2. **Nucleus detection** by absence of signal: dark voxels (below the 30th percentile
   of the cell), the largest and most central dark region, one body per optical plane,
   and 3D smoothing (0.285 µm).
3. **Inclusion gates**: border contact, cell volume (400–6000 µm³), receptor and marker
   saturation, EGFP-positive fraction of the cytoplasm (≥ 0.13), spatial spread of the
   EGFP signal (block coverage ≥ 0.61), binucleate cells, and a marker gate specific to
   each compartment.
4. **Compartment mask** (table above).
5. **Enrichment**, one metric for the four compartments:

   ```
   D = median(EGFP in the mask) / median(EGFP in the reference)
   reference = cell − nucleus − inner dead band (0.85 µm) − mask
   ```

   D = 1: same concentration as the rest of the cell; D > 1: enriched; D < 1: depleted.

6. **Per-cell null (Costes block randomisation).** Receptor-channel blocks the size of
   the point-spread function (2 × 1 × 1 voxels) are shuffled inside mask + reference,
   with the masks held fixed, 200 times. A cell is called *enriched* above the 97.5th
   percentile of its own null and *depleted* below the 2.5th.

All thresholds were set from the distribution of the data or from blind visual
assessment; they are listed in the `CFG` dictionary at the top of the script.

## Installation

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for the Cellpose step. Install the PyTorch
build that matches your CUDA version from https://pytorch.org if the default one does
not fit your system. Once the label caches exist, the rest of the pipeline runs on CPU.

## Usage

```bash
python vpac_enrichment.py <folder | file.oib> [out_dir] \
    --comp caax|fyve|golgi|emerin --receptor VPAC1|VPAC2 \
    [--use-labels] [--dz 0.35] [--no-null]
```

| Option | Meaning |
|---|---|
| `--comp` | Compartment to analyse (sets segmentation channel, mask and marker gate) |
| `--receptor` | Label written to the output table |
| `--use-labels` | Reuse cached segmentations (`*_labels.npz`) instead of running Cellpose |
| `--dz` | Force the Z step in µm if the metadata are missing or wrong |
| `--no-null` | Skip the Costes null (faster; no per-cell verdict) |

A relative `out_dir` is created under `$COLOC_BASE` (default `~/Colocalizacion`).

### File naming

Conditions and field numbers are read from the file name:

| File name | Condition |
|---|---|
| `CAAX1.oib` | basal, field 1 |
| `CAAXV303.oib` | VIP 30 min, field 3 |
| `FYVEAV1152.oib` | selective agonist (AV1) 15 min, field 2 |
| `EMERINAV2901.oib` | selective agonist (AV2) 90 min, field 1 |

Prefixes: `CAAX`, `FYVE`, `B3GALT` (B4GALT1 construct), `EMERIN`.

> Note on the thesis dataset: files labelled `60P` (e.g. `CAAXV60P1`) correspond to
> 90 min of stimulation.

## Output

One folder per run:

| File | Content |
|---|---|
| `results_<comp>_cells.csv` | One row per segmented cell |
| `results_<comp>_cells.xlsx` | Same table plus a sheet describing every column |
| `qc/` | Per-field overlays of included/excluded cells and masks |

Main columns: `included`, `reason` (gates failed), `enrichment_D` (primary metric),
`null_p2_5`, `null_p97_5`, `null_verdict` (`enriched` / `depleted` / `ns`) and
`null_mask_fraction` (must be ~1.0 for the verdict to be valid).

For statistics, the field (image + acquisition session) should be used as the unit of
analysis, summarised as the median D of its included cells.

## Companion script

`vpac_methods_figures.py` draws the nucleus-detection and mask-construction steps for a
single cell and checks that every intermediate step reproduces the pipeline output.

## Limitations

- The nucleus is detected by absence of signal (no nuclear stain); the detector always
  returns a body, so segmentation errors are not flagged automatically.
- The nuclear-envelope shell (0.30 µm) and the Golgi cisternae are below the axial
  resolution of a confocal microscope; masks describe regions, not organelle membranes.
- Thresholds were calibrated on this dataset (HEK-derived overexpression, Olympus
  FV1000, 0.207 µm/px, 0.35–0.50 µm Z step) and may need recalibration elsewhere.

## Use of AI

The code was developed with the assistance of Claude (Anthropic). Methodological
decisions, thresholds and validation against visual inspection of the images were made
by the authors.

## Citation

If you use this code, please cite the thesis above and this repository
(Zenodo DOI: [10.5281/zenodo.XXXXXXX]).

Key methods: Stringer et al., *Nat Methods* 2021 (Cellpose); Costes et al., *Biophys J*
2004 (block randomisation).

## License

MIT (see `LICENSE`).
