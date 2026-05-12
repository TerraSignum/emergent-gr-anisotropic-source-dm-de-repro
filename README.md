# emergent-gr-anisotropic-source-dm-de-repro

[![CI: reproduce](https://github.com/TerraSignum/emergent-gr-anisotropic-source-dm-de-repro/actions/workflows/reproduce.yml/badge.svg)](https://github.com/TerraSignum/emergent-gr-anisotropic-source-dm-de-repro/actions/workflows/reproduce.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Reproducibility package for the **anisotropic source tensor** and the
**dark-matter / dark-energy (DM/DE) mixture decomposition** on the
relational emergent-gravity construction. Topic-focused companion
repository to `emergent-gr-closure-repro` (Paper 4) which reports the
per-node 4×4 Galerkin closure and the structural cosmological-tensor
identification.

## What this repository contains

The relational lattice carries a structured anisotropic source tensor
$T_{\mu\nu}^{\Xi}$ obtained by Hilbert variation of the carrier
action on the spectral basis. Three reproducible certificates are
bundled:

1. **Energy-condition diagnostics:** Bundled NEC/SEC/DEC tests on the
   nine-regime ladder; the asymptotic-window mean reads
   $\rho + p \approx 0$ and $\rho + 3p \approx +0.11$.

2. **DM/DE mixture decomposition:** The diagonal-block source
   decomposes into a dark-matter-vortex component (winding-number
   topological defects, $w_{\text{DM}} = 0$) and a dark-energy
   non-scalar Clifford-channel component
   ($w_{\text{DE}} = -1 + \varepsilon^{4}_{\text{sync}}/\gamma$).
   The pooled effective EOS is $w_{\text{eff}} \approx -0.31$,
   matching the cosmic-string-network analytical $w = -1/3$ within
   seed-noise envelope.

3. **Cosmological-constant eight-layer dressing (preliminary):** A
   parameter-free numerical exercise; the construction reduces the
   Planck-scale ↔ observed cosmological-constant hierarchy by 122
   orders of magnitude (ratio 0.89, residual $-0.05$ OoM, sub-decimal
   closure on the canonical regime). Reported as preliminary, not
   principal.

4. **Inflation and vacuum-stability readouts (preliminary):** Spectral
   tilt $n_{s} = 1 - \gamma\,\varepsilon^{2}_{\text{sync}}$,
   tensor-to-scalar ratio $r = 0.029$, vacuum-decay action
   $B \to \infty$ (absolutely stable on the canonical regime).

## Tier-1 reproduction (frozen JSON, no compute)

```bash
pip install -e .
pytest tests/
```

Expected: 23 unit tests pass.

## Tier-2 reproduction (D1 lattice run, optional)

The energy-condition diagnostics and the DM/DE mixture
decomposition at higher resolution require the D1 lattice NPZ files
which are not bundled. With these in `data/d1_runs/`, the GPU
verification scripts can be re-run.

## Companion paper structure

This repo extracts the anisotropic-source / DM-DE / cosmology
material from the parent emergent-gravity closure paper (P4) and
presents it as a focused topic-paper. The parent retains the per-node
4×4 Galerkin closure as its principal result.

## License

MIT.