# LPA v1.0.0 release notes

LPA v1.0.0 is the first public release of the Label Poisoning Antidote
architecture, released under the MIT License.

The architecture is the one frozen in 1.0.0rc1 after 33 research and
qualification gates, and produces bit-identical models. The core is
intentionally smaller than the research tree. Proprietary VIVERE and MACSL
components were removed; geometry-based poison detectors, graph diffusion,
mandatory conformal projection, confidence filtering, calibration rewrites,
and smarter landmark selection were all rejected from the default path after
failing to provide reliable utility gains.

The release keeps the pieces that survived repeated stress testing:
trusted-only multiview teaching, direct soft pseudo-targets, x-only random RBF
landmarks, ridge distillation, and exact sufficient-statistic federation.

## Since 1.0.0rc1

- MIT license.
- Every array in a saved model is integrity-checked on load, with an optional
  out-of-band fingerprint for tamper evidence.
- Invalid inputs (non-integer labels, NaN/inf features, wrong feature widths,
  malformed federated statistics) now fail loudly instead of silently.
- Cross-platform CI, linting, type checking, and wheel verification.

See [`CHANGELOG.md`](CHANGELOG.md) for details.

## Claims

Full-data qualification evidence is bundled in `benchmarks/results/`. These
results have not yet been independently reproduced.

LPA is intentionally conservative about claims. It guarantees direct non-use
of an explicitly untrusted label field in the supplied implementation; it does
not claim recovery of the latent true label or robustness to threats outside
the documented model. See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).
