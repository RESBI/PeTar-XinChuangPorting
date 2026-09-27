# LA664 (Loongson-3A6000) LSX/LASX tree-force kernels -- benchmark and report

This directory contains the port report for the 128-bit LSX and 256-bit LASX
tree-force kernels of PeTar (`src/force_loongarch.hpp`), selected at build time by:

```shell
./configure --with-arch=loongarch --with-simd=lasx    # or: lsx / no
```

The vector width can also be overridden at the make stage, e.g. `make LARCH_SIMD=lsx`
(or `make LARCH_SIMD=none`); the executable name follows the configure-time choice
(`petar.mpi.omp.lasx`, `petar.mpi.omp.lsx`, or `petar.mpi.omp`).

Contents:

- `REPORT.md` -- full report (English): port design, four-stage numerical
  validation, kernel and end-to-end benchmarks, scaling, per-step breakdown,
  conservation, production run, and cross-platform positioning.
- `REPORT_CN.md` -- Chinese version of the report.
- `figs/` -- figures for the scalar / LSX / LASX builds.
- `figs_cross/` -- cross-platform comparison and methodology figures.
- No raw benchmark data are stored in this repository; every figure and table
  in the reports is derived from the measurements described in them.

Key results on the 3A6000 (4 physical cores x 2 SMT, 2.5 GHz measured at 2.50 GHz, GCC 15.3):

| workload | scalar (F64) | LSX (128-bit) | LASX (256-bit) |
|---|---|---|---|
| EP-EP kernel [ns/interaction] | 8.99 | 2.80 (3.2x) | **1.42 (6.3x)** |
| single core, N=2000 demo, 512 steps [s] | 38.86 | 18.70 (2.08x) | **14.10 (2.76x)** |
| best layout (1 process x 4 OMP threads) [s] | 12.49 | 6.04 | **4.53** |
| N=8000, 4x1, per step [ms] | 213.5 | 85.7 | **57.1** |
| 100 Myr production sample, 8 OMP threads | 28 min 16 s | **10 min 54 s** | 11 min 39 s |

Numerical validation against the scalar reference (`petar.simd.test` style):
max relative force error 4.3e-4 (tolerance 7e-3), neighbor counts identical
particle by particle.
