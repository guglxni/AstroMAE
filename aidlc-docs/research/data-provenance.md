# Data Provenance (RESEARCH-07)

## Galaxy10 SDSS (existing, optional)

- **Source**: astroNN — Leung & Bovy 2019.
  [`load_galaxy10`](https://astronn.readthedocs.io/en/latest/galaxy10.html)
  downloads from the astroNN mirror.
- **Licence**: SDSS data is public domain for astronomical research;
  astroNN wrappers are MIT-licensed.
- **Derivation**: 3-channel RGB PNGs at 69×69 px.
- **Use in this repo**: optional downloader under
  `astromae.data.download.download_and_save_galaxy10`. Not committed.
- **Limitations**: RGB-only, no IVAR, no multi-band. **Not suitable for
  claims C1 (IVAR-weighted loss)**; retained only as a morphology
  classification baseline.

## Synthetic FITS cutouts (new, always available)

- **Source**: generated in-repo by
  `astromae.data.synth.make_synthetic_fits` (M1.4).
- **Licence**: code is MIT; data never persisted beyond user's
  workspace.
- **Derivation**: Sérsic-profile galaxies with controllable PSF,
  Poisson + Gaussian noise, two-survey profiles (A, B) for
  cross-survey pilot experiments. Includes exact IVAR maps.
- **Use in this repo**: drives all CI unit tests for FITS, IVAR-weighted
  loss, equivariance, and cross-survey protocol.

## DECaLS-SV FITS cutouts (planned, real data)

- **Source**: Legacy Survey cutout service —
  `https://www.legacysurvey.org/viewer/fits-cutout` — public,
  scriptable, supports multi-band (g, r, z) and IVAR planes.
- **Licence**: Legacy Surveys data are released under CC-0 / public
  domain; citation required: Dey et al. 2019 + survey acknowledgement.
- **Derivation**: 1000-galaxy DR10 pilot sample with matched
  spectroscopic redshift from DESI-EDR (Zhou et al. 2024 public
  tables) for photo-z pilot.
- **Use in this repo**: provisioned via a downloader script
  (`scripts/download_decals_pilot.py`, not in this PR — deferred to
  first real-data run). Data itself is gitignored.

## HSC-PDR public subset (planned, real data, cross-survey partner)

- **Source**: HSC-SSP Public Data Releases.
- **Licence**: permissive academic use with citation (Aihara et al.).
- **Use**: the "Survey B" half of the cross-survey pilot once the
  DECaLS side works end-to-end.

## Euclid DR1 (not used, noted for completeness)

- Walmsley's Euclid-MAE uses this. We do not replicate because the
  Euclid-MAE model and 13.6M training sample are already public on
  Hugging Face. We would use the *released Walmsley weights* as a
  comparison baseline only, not re-train.

## Provenance integrity

Each downloader script will write a `manifest.json` listing, per file:

```json
{
  "filename": "dr10_cutout_000123.fits",
  "source_url": "...",
  "sha256": "...",
  "retrieved_at": "2026-04-20T14:00:00Z",
  "license": "CC-0",
  "citation": "Dey et al. 2019 AJ 157, 168"
}
```

CI runs against synthetic data only; real-data sections of the
evaluation matrix are gated on a developer machine with the manifest
present.
