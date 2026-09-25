# Licences of the training environment

`requirements-train.licences.json` records the licence of every package in
`requirements-train.lock`: 31 packages, installed with `--require-hashes` on
Python 3.13, linux x86_64. For each package it records:
- its SPDX licence;
- what the wheel declares;
- the sha256 and size of every licence text the wheel ships;
- bundled components and vendored shared libraries.

The sentencepiece wheel ships no licence text, so its five texts come from
the hash-pinned 0.2.2 sdist.

- **Top-level licences.** Every package is under a permissive licence:
  Apache-2.0, MIT, BSD-2/3-Clause, PSF-2.0, 0BSD, Zlib or CC0-1.0.
- **Bundled terms beyond attribution:**
  - numpy and scipy vendor `libgfortran` (GPL-3.0-or-later WITH
    GCC-exception-3.1) and `libquadmath` (LGPL-2.1-or-later);
  - jaxlib includes Eigen (MPL-2.0).
- **Nothing here is redistributed.** Only the lock ships. The machine that
  tunes downloads the wheels from PyPI and checks them against the lock's
  hashes.
- **If a release ever bundles the wheels,** it must:
  - ship the licence texts;
  - offer the LGPL source for libquadmath;
  - keep Eigen's MPL notice.
- **Trained weights** contain none of this code, so these licences don't reach
  them.

When the lock changes, regenerate the records from a fresh install, then run
`tests/test_train_licences.py`: it fails while the lock and the records
disagree.
