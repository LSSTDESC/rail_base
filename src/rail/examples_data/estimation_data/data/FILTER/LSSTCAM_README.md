## lsstcam_{u,g,r,i,z,y}.res — LSST DP2 standard passbands

These are the standard passbands used for the Rubin/LSST Data Preview 2 (DP2)
photometric calibration.  Each file contains two columns: wavelength [Angstrom]
and total throughput [0–100].

### What is included

The standard passband is a total system throughput curve that folds together:

- **Atmosphere** — standard LSST site atmosphere (Cerro Pachón, 1.2 airmasses)
- **Filter** — bandpass filter transmission
- **Detector QE** — a combination of the ITL and E2V sensor quantum efficiency
  curves (both vendors were used for the LSSTCam focal plane)
- **Optics** — all three lenses (L1, L2, L3) and the mirror reflectivity

This is the curve that all DP2 photometry is calibrated to (i.e. the "standard"
in the FGCM forward-modelling calibration), so it is the appropriate passband
to use when computing synthetic photometry from SEDs for photo-z or SED fitting.

### Source

Retrieved from the `dp2_prep` Butler repository, dataset type `standard_passband`,
collection `LSSTCam/runs/DRP/DP2`, one dataset per band.
