# Changelog — `emergent-gr-anisotropic-source-dm-de-repro`

P4B anisotropic source / DM-DE / cosmological-constant.

## v1.0.0 — 2026-05-12 (publication-prep freeze)

- Frozen publication-ready state after the multi-round
  review/optimization autopilot of 2026-05-11/12.
- Cross-corpus invariants verified: tests pass, manuscripts
  compile, zero unresolved citations or references, SHA256 of
  bundled data files matches `data/SHA256SUMS`.
- See `RELEASE_v1_0_0.md` for the per-repo headline state and
  `compendium/CORPUS_STATE_2026_05_12.md` for the corpus-wide
  publication-prep summary.

## v0.3.0 — 2026-05-12 (H_sync promotion)

- Cosmological-constant nine-layer dressing promoted from
  8-layer factor-2 PRECISE (ratio 0.89) to 9-layer EXACT
  (ratio 1.001) via the H_sync layer
  (synchronization-channel un-cancellation, derived from
  P2 §5 fluctuation-dissipation identity C_3:
  eps_sync^2 = gamma/2 = 1/20).
- Closure residual: +0.0004 OoM (0.1% above Planck 2018).
- `verify_cosmological_constant.py` reproducibility check
  hardened: now recomputes ratio from per-layer log10_contribution
  + M_Pl^4 and asserts match within 0.02 OoM (test_repro_match).
- Pre-registration block stamped at SHA-256
  `aa563454e9ff5404...` (5 near-term cosmology predictions).

## v0.2.1 — 2026-05-11 (H197 retraction)

- Previous H197 paired-degeneracy correction retracted from
  the published closure. The layer was sign-inconsistent
  across the code path (production GCC-07 applied +0.32 OoM,
  System-R identification asserted -0.33 OoM) and had no
  external QFT analogue. Retraction recorded in the
  bundled `h197_retraction_2026_05_11` block.

## v0.2.0 — 2026-05-10 (initial cosmology paper)

- Initial publication-prep state with 8-layer CC closure at
  ratio 1.83 (factor-2 PRECISE tier).
