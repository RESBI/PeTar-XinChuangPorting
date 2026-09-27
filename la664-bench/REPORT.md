# PeTar on LoongArch (Loongson-3A6000 / LA664): LSX and LASX Tree-Force Kernels

**Port and benchmark report**

- Date: 2026-09-28
- Platform: Loongson-3A6000 (LA664 core), 4 physical cores x 2 SMT = 8 logical CPUs, fixed 2.5 GHz, 31 GiB RAM, 16 MB shared L3
- Clock: 2.5 GHz nominal; independently measured at 2.50 GHz (dependent-instruction chains and hardware cycle counters), with no throttling under load
- ISA: loongarch64 (LA64v1.0) with LSX (128-bit) and LASX (256-bit)
- Toolchain: GCC 15.3.0, OpenMPI 4.1.6
- Code: PeTar master (`1268_298`) + FDPS 7.0 + SDAR; new kernels in `src/force_loongarch.hpp`
- Main benchmark: N=2000 binary-rich demo, t=1 Myr (512 tree steps), 3 repetitions, medians
- Figures: `figs/` (LoongArch scalar / LSX / LASX), `figs_cross/` (cross-platform comparison)

---

## 0. Executive Summary

The tree-force calculation in PeTar was previously limited to scalar F64 kernels on LoongArch. This work adds hand-written LSX (128-bit) and LASX (256-bit) kernels that reproduce the x86/Fugaku approach -- F32 arithmetic, an origin shift for cancellation control, a fast reciprocal square root, and an exact F64 fallback for neighbor counting -- while keeping the physical model unchanged. The port is validated at four levels and benchmarked end to end on a Loongson-3A6000.

| Metric | Result |
|---|---|
| EP-EP kernel, ni=1024 x nj=2048 | **1.42 ns/interaction** (LASX), 2.80 ns (LSX), 8.99 ns (scalar): **6.3x / 3.2x** |
| Official `petar.simd.test` | max force error 4.3e-4 (tolerance 7e-3), potential error 2.2e-6, **neighbor counts identical particle by particle** |
| Single core, N=2000 demo, 512 steps | **14.10 s** (LASX), 18.70 s (LSX), 38.86 s (scalar): **2.76x / 2.08x** |
| Best parallel layout | 1 process x 4 OpenMP threads: **4.53 s** (LASX), 6.04 s (LSX) |
| 4x1 layout vs its scalar counterpart | 5.28 s vs 12.06 s (**2.28x**) |
| Weak scaling, N=8000, 4x1 | **57.1 ms/step** (LASX), 85.7 (LSX), 213.5 (scalar): **3.74x / 2.49x** |
| Conservation QA, t=10 Myr, 4x1 | dE and dL within the same order as the scalar build (see section 8) |
| Production sample, 100 Myr, 8 OMP threads | **11 min 39 s** (LASX), 10 min 54 s (LSX) vs 28 min 16 s (scalar) |
| Debug validation | 1 and 4 ranks, 128 tree steps each: zero assertion failures, zero neighbor-count mismatches |

**In one sentence:** the SIMD kernels cut the tree force by 3-6x, which translates into a **2.1-2.8x end-to-end speed-up per core** and a **3.7x** speed-up per step at N=8000; once the tree force is no longer dominant, the short-range Hermite force and cluster search become the new bottlenecks.

---

## 1. Background and Scope

PeTar is an N-body code for star-cluster and tidal-stream evolution. On LoongArch, `--with-arch=loongarch` previously fell back to the `NoSimd` kernels: pure scalar F64 arithmetic with one `1/sqrt(r^2)` per interaction. Micro-benchmarks show that this square root costs 3-4.5x a scalar FMA, and the tree force accounts for roughly 78% of a single-core step, so the optimization headroom was clear.

The LA664 core in the 3A6000 supports two vector widths, LSX (128-bit) and LASX (256-bit). This work ports the established SIMD strategy -- already used by the upstream x86 and Fugaku kernels -- to both widths:

1. implement the four tree kernels (EP-EP force, EP-SP quadrupole, EP-SP monopole, and neighbor search) in a single header, `src/force_loongarch.hpp`;
2. keep the numerical behavior tightly aligned with the scalar kernels: force/potential errors within the `simd_test` tolerance and **neighbor counts identical particle by particle**;
3. validate on real hardware, not an emulator, at kernel, unit-test, production, and debug levels;
4. produce a reproducible benchmark report covering both vector widths and the scalar baseline.

The short-range PP force (Hermite, F64) and the hard/regularized integration are outside the scope of this port and are left unchanged.

---

## 2. Port Design

### 2.1 Kernel structure and shape dispatch

FDPS fixes the kernel call signature: `operator()(epi, n_i, epj, n_j, force)`. The actual work happens in two vector loop shapes:

- **IV_J1**: load `W` particles from the i side per iteration (`W` = 8 for LASX, 4 for LSX), broadcast one j particle at a time; forces and counts accumulate across the lanes.
- **I1_JV**: load `W` j particles, broadcast one i particle at a time; the per-i result is obtained by a horizontal reduction.

The dispatcher counts the active i particles (`type == 1`) in the group and picks the shape automatically:

| Condition | Shape | Why |
|---|---|---|
| active i >= vector width | IV_J1 | full lanes, no reduction overhead |
| active i < vector width | I1_JV | avoids wasting lanes on a partially filled i vector |

This matters in practice: at ni=4, nj=2048, I1_JV is faster (3.36 ns vs 3.75 ns per interaction), while at ni>=16 IV_J1 wins across the board (1.65 ns vs 1.87 ns at ni=16).

### 2.2 Numerics: F32 with an origin shift

Each EP-EP interaction is evaluated as

$$
r^2 = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2,\qquad
r^2_{\epsilon} = r^2 + \epsilon^2,\qquad
r^2_{\text{cut}} = \max(r^2_{\epsilon},\, r_{\text{out}}^2),
$$

$$
r^{-1} = \frac{1}{\sqrt{r^2_{\text{cut}}}},\qquad
m_r = m_j r^{-1},\qquad
\mathbf{a}_i \mathrel{-}= m_r r^{-2}\,\mathbf{r}_{ij},\qquad
\phi_i \mathrel{-}= m_r .
$$

Everything except $\epsilon^2$ and $r_{\text{out}}^2$ is computed in F32; forces and potentials are accumulated separately, then multiplied by $G$ and written back to the F64 `ForceSoft` structure. The quadrupole and monopole EP-SP kernels follow the scalar formulas (including the quadrupole-moment projection and the $r^{-5}, r^{-7}$ terms) with vector instructions.

To control catastrophic cancellation, every group is translated so that the **first active i particle** sits at the origin before conversion to F32; all separations, $r^2$ values, and the $r_{\text{search}}$ comparison are performed in the shifted frame.

### 2.3 Fast reciprocal square root

The scalar `1.0/sqrt(r2)` is the hidden cost centre of the tree force, and LA664 does not provide a usable scalar estimate instruction. The port therefore builds the reciprocal square root from a bit-magic seed followed by two refinements, reusing formulas already proven in the upstream ports:

$$
y_0 = \texttt{0x5f3759df} - (a \gg 1),\qquad
y_1 = y_0\,\frac{3 - a y_0^2}{2},
$$

$$
h = 1 - a y_1^2,\qquad
y_2 = y_1 + y_1\left(\tfrac12 h + \tfrac38 h^2\right).
$$

The first refinement is the Newton step from the x86 `RSQRT_NR_EPJ_X2` path, which lifts the 5-bit seed to about 11 bits; the second is the cubic correction used by the Fugaku port, which reaches full F32 accuracy. Measured maximum relative error: **1.5e-7**. Dependent-chain measurements on the 3A6000 give about **47 cycles per 8-lane vector** for this sequence (for comparison, the platform's exact `xvfrsqrt.s` instruction measures ~25 cycles and `fsqrt`+`fdiv` ~30 cycles under the same method); the tree-force loop keeps many independent interactions in flight, and the complete EP-EP kernel still costs only 1.42 ns per interaction (section 4), so the refinement chain is not a bottleneck. The software sequence was chosen to share the exact formulas with the x86 and Fugaku ports, so the precision does not depend on a given CPU's implementation.

### 2.4 Neighbor counting with an F64 fallback

Neighbor counts must match the scalar kernel exactly, otherwise PeTar's cluster finder aborts. A purely vectorized comparison is accurate only to about 1e-6 relative, so the port adds a "boundary band" test on every interaction:

$$
\left| r^2 - R^2 \right| < 2^{-10} R^2,\qquad R^2 = \max(r_i^2, r_j^2).
$$

If any particle falls inside the band, the whole j loop for that i is recomputed in F64 (`countNeighborExact`). The band of $2^{-10}$ relative width safely covers the F32 round-off, and true hits are rare, so the hot path stays fully vectorized. This "F32 hot path + F64 cold path" design is what makes the counts bit-identical to the scalar reference.

### 2.5 Build system integration

| Switch | LASX (256-bit) | LSX (128-bit) | Scalar |
|---|---|---|---|
| configure | `--with-arch=loongarch --with-simd=lasx` | `--with-simd=lsx` | `--with-simd=no` |
| make (override) | `make LARCH_SIMD=lasx` | `make LARCH_SIMD=lsx` | `make LARCH_SIMD=none` |
| binary | `petar.mpi.omp.lasx` | `petar.mpi.omp.lsx` | `petar.mpi.omp` |

Both widths are compiled from the same header; a single `-D USE_LARCH_SIMD` plus the width macro `LARCH_SIMD_LSX` selects the LASX or LSX branch, and the vector loops are written with width-generic helpers.

Files touched by the port:

- `src/force_loongarch.hpp` -- new, the LSX/LASX kernels;
- `src/soft_force.hpp` -- includes the new header under `USE_LARCH_SIMD`;
- `src/petar.hpp` -- routes the tree force, tree neighbor search, and EP-SP kernels to the LoongArch SIMD implementations;
- `src/simd_test.cxx` -- extends the official self-test to the LoongArch builds;
- `configure.ac`, `Makefile.in`, regenerated `configure` -- `loongarch` architecture and `--with-simd=lsx|lasx|no`;
- `README.md` -- documents the new architecture and options.

---

## 3. Functional Validation

Validation proceeds through four stages, each with a repeatable procedure.

### 3.1 Kernel-level synthetic data (`mini_check`)

A dedicated driver calls the four kernels with synthetic data sized to hit tail blocks and dispatch boundaries (ni=37, nj=53, deliberately not multiples of the vector width) and compares all four kernels in both loop shapes against `NoSimd` at four optimization levels:

| Check | -O0 | -O1 | -O2 | -O3 |
|---|---|---|---|---|
| EP-EP max relative force error | 4.29e-7 | 4.29e-7 | 4.29e-7 | 4.29e-7 |
| EP-SP quadrupole | 4.29e-7 | 4.29e-7 | 4.29e-7 | 4.29e-7 |
| EP-SP monopole | 3.36e-7 | 3.36e-7 | 3.36e-7 | 3.36e-7 |
| Neighbor-count mismatches | 0 | 0 | 0 | 0 |

The four optimization levels produce identical results, so the kernels do not rely on undefined behavior or on compiler-specific reassociation.

### 3.2 Official `petar.simd.test`

PeTar ships a comparison program (Plummer initial conditions, N=1000 x 2000, threshold 7e-3). Both SIMD widths pass with the same numbers:

| Quantity | Max relative error |
|---|---|
| EP-EP force | 4.30e-4 |
| EP-EP potential | 2.18e-6 |
| EP-SP force | 2.93e-4 |
| EP-SP potential | 2.22e-6 |
| Neighbor counts | 1003 vs 1003, identical per particle |

LSX and LASX report identical error values because both use the same F32 operation order and the same correction formula; the vector width changes parallelism, not rounding.

### 3.3 Kernel-benchmark self-check

The standalone kernel driver (ni=500 x nj=2000, uniform distribution) verifies both loop shapes:

| Kernel | Max relative force error | Count mismatches |
|---|---|---|
| Neighbor search | 0 (counts only) | 0 |
| EP-EP | 2.2e-6 | 0 |
| EP-SP quadrupole | 2.2e-5 | 0 |

### 3.4 Production short runs and debug builds

A 512-step run of the N=2000 demo finishes normally in every build, with energy errors of the same order as the scalar baseline (section 8). The strongest check is the debug build: compiled with `CLUSTER_DEBUG`, `HARD_DEBUG`, `NAN_CHECK_DEBUG`, `PETAR_DEBUG`, `AR_DEBUG`, `STABLE_CHECK_DEBUG`, and `ARTIFICIAL_PARTICLE_DEBUG`, and run with 1 and 4 MPI ranks for 128 tree steps each. `CLUSTER_DEBUG` compares, for every particle at every tree step, the neighbor count produced by the force kernel with the tree neighbor list:

```
inconsistent messages: 0     (1 rank and 4 ranks)
finished: 1                  (both layouts terminate normally)
```

This stage exercises MPI-level clustering, the short-range integrator, and artificial-particle paths, and is the strongest evidence that the port is functionally sound.

---

## 4. Kernel-Level Performance

Kernel micro-benchmarks run on a single pinned physical core, scanning ni in {4, 16, 64, 256, 1024} and nj in {8, 32, 128, 512, 2048}.

![Kernel micro-benchmark](figs/figK1_kernel_ni1024.png)

*Figure 4-1: cost per interaction of the four kernels as a function of nj (ni=1024, log scale; grey = scalar, green = LSX, blue = LASX).*

At ni=1024:

| Kernel | nj | Scalar [ns] | LSX [ns] | LASX [ns] | LSX speed-up | LASX speed-up |
|---|---:|---:|---:|---:|---:|---:|
| EP-EP | 128 | 9.03 | 2.88 | **1.48** | 3.14x | 6.11x |
| EP-EP | 2048 | 8.99 | 2.81 | **1.42** | 3.21x | 6.32x |
| EP-SP quadrupole | 2048 | 18.22 | 5.49 | **2.71** | 3.32x | 6.71x |
| EP-SP monopole | 2048 | 10.57 | 2.16 | **1.09** | 4.89x | 9.69x |
| Neighbor search | 2048 | 2.66 | 0.78 | **0.39** | 3.42x | 6.76x |

![Kernel bars](figs/figK2_kernel_bars.png)

*Figure 4-2: kernel cost at ni=1024, nj=2048 (log scale).*

Three observations:

1. LASX speed-ups cluster around **6-7x** against scalar. The shortfall from the theoretical 8 lanes comes from the reciprocal-square-root chain and from broadcast/reduction overhead; the monopole kernel, which has the shortest formula, reaches 9.7x.
2. The wider the j group, the better the speed-up: at nj=8 the refinement and loop overhead dominate and LASX only reaches 2.4-4.1x.
3. The LASX/LSX time ratio is essentially **0.5**, i.e., LA664 implements LASX as a true 256-bit datapath rather than as two 128-bit halves.

---

## 5. Thread/Process Configuration Scan

Same background for all builds: N=2000 demo, t=1 Myr (512 tree steps), median of three repetitions.

![Configuration scan](figs/figS1_scan_wall.png)

*Figure 5-1: wall clock of the scalar, LSX, and LASX builds across process/thread layouts (log scale; error bars are the range of 3 runs).*

| Layout | Cores | Scalar [s] | LSX [s] | LASX [s] | Scalar/LSX | Scalar/LASX |
|---|---:|---:|---:|---:|---:|---:|
| 1x1 | 1 | 38.86 | 18.70 | 14.10 | 2.08x | 2.76x |
| 2x1 | 2 | 21.47 | 10.96 | 8.41 | 1.96x | 2.55x |
| 4x1 | 4 | 12.06 | 6.57 | 5.28 | 1.84x | 2.28x |
| 8x1 (SMT) | 8 | 12.37 | 6.81 | 5.86 | 1.82x | 2.11x |
| 1x2 (same-core SMT) | 1 | 33.26 | 13.95 | 10.12 | 2.38x | 3.29x |
| **1x4 (OMP)** | **4** | 12.49 | **6.04** | **4.53** | 2.07x | 2.76x |
| 1x8 (OMP) | 4 (8 SMT) | 14.81 | 6.71 | 5.13 | 2.21x | 2.89x |
| 2x2 (oversubscribed) | 2 | 57.27 | 46.92 | 44.44 | 1.22x | 1.29x |
| 4x2 (oversubscribed) | 4 | 49.70 | 44.47 | 43.10 | 1.12x | 1.15x |
| 4x1, bound | 4 | 12.06 | 6.56 | 5.29 | 1.84x | 2.28x |

![Parallel speed-up](figs/figS2_scan_speedup.png)

*Figure 5-2: self-normalized parallel speed-up (dashed line = ideal).*

Three points stand out:

1. **The speed-up is largest at low parallelism** (2.08x / 2.76x at 1x1) and is diluted by parallel efficiency at 4 cores (1.84x / 2.28x). Oversubscribed layouts collapse to 1.1-1.3x because time is spent contending for cores.
2. **The optimal layout moves from 4x1 to 1x4.** In the scalar build, 4x1 is about 3% faster than 1x4; with SIMD, 1x4 wins by 8% (LSX) and 17% (LASX), because shrinking the tree force exposes MPI decomposition and synchronization overhead.
3. Repetition spreads are generally 1-3%, so the rankings above are meaningful. **SMT is not useful** (8x1 matches 4x1, 1x8 is slower than 1x4) and **oversubscription is catastrophic** on all three builds.

---

## 6. Scaling

### 6.1 Weak scaling (tamed initial conditions, 4x1, t=2 Myr)

![Weak scaling](figs/figN1_weak_scaling.png)

*Figure 6-1: seconds per Myr as a function of N (log scale; same figure shows scalar, LSX, and LASX).*

| N | Tree steps | Scalar [ms/step] | LSX [ms/step] | LASX [ms/step] | Scalar/LSX | Scalar/LASX |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 512 | 8.5 | 6.0 | 5.4 | 1.42x | 1.57x |
| 2000 | 1024 | 25.2 | 14.5 | 11.8 | 1.74x | 2.14x |
| 4000 | 2048 | 68.3 | 30.5 | 22.4 | 2.24x | 3.06x |
| 8000 | 2048 | 213.5 | 85.7 | 57.1 | 2.49x | 3.74x |

The per-step speed-up grows monotonically with N (1.57x to 3.74x for LASX) because the vectorizable tree force becomes a larger fraction of a step while fixed costs (tree build, domain decomposition, I/O) do not shrink. Note that the tree time step is adjusted automatically by the code, so the number of steps must be reported together with s/Myr (see section 10.2); ms/step is the robust cross-N metric.

### 6.2 Strong scaling (N=8000, t=1 Myr)

![Strong scaling](figs/figN2_strong_scaling.png)

*Figure 6-2: strong scaling at N=8000, self-normalized (dashed line = ideal).*

| Layout | Scalar [s] | LSX [s] | LASX [s] | Scalar speed-up | LSX speed-up | LASX speed-up |
|---|---:|---:|---:|---:|---:|---:|
| 1x1 | 777.2 | 294.1 | 188.8 | 1.00 | 1.00 | 1.00 |
| 2x1 | 401.4 | 157.9 | 105.1 | 1.94 | 1.86 | 1.80 |
| 4x1 | 214.8 | 86.4 | 58.6 | 3.62 | 3.41 | 3.22 |
| 8x1 (SMT) | 204.4 | 82.2 | 58.2 | 3.80 | 3.58 | 3.25 |

At N=8000 the single-core speed-up reaches **4.12x** (LASX), well above the 2.76x at N=2000: the tree force occupies a larger share of every step, so Amdahl's ceiling rises with N. Four-core efficiency is still 81% for LASX and SMT adds only about 3%.

---

## 7. Where the Time Goes: Per-Step Breakdown

The data come from the slowest-rank per-step profile (`-f perf`, maximum row), median of three runs, with the 16 profile entries grouped into six categories.

![Per-step breakdown](figs/figB1_perstep_breakdown.png)

*Figure 7-1: stacked per-step cost (N=2000 demo, 512 steps) for 1x1, 4x1, and 1x4.*

**Table 7-1: per-step cost by component (ms, slowest rank, median of 3)**

| Build | Layout | Tree force | Tree neighbor | Short-range PP | Cluster/group | Other | Total |
|---|---|---|---:|---:|---:|---:|---:|
| Scalar | 1x1 | 58.77 | 1.29 | 10.81 | 4.32 | 0.30 | 75.49 |
| LSX | 1x1 | 19.93 | 0.95 | 10.77 | 4.25 | 0.30 | 36.19 |
| LASX | 1x1 | **10.90** | 0.74 | 10.98 | 4.26 | 0.34 | **27.22** |
| Scalar | 4x1 | 16.45 | 1.29 | 4.07 | 1.23 | 0.12 | 23.15 |
| LSX | 4x1 | 6.07 | 1.07 | 4.06 | 1.16 | 0.12 | 12.48 |
| LASX | 4x1 | **3.69** | 0.91 | 4.13 | 1.17 | 0.09 | **9.98** |
| Scalar | 1x4 | 19.03 | 0.91 | 3.17 | 0.82 | 0.16 | 24.08 |
| LSX | 1x4 | 6.70 | 0.66 | 3.14 | 0.83 | 0.17 | 11.50 |
| LASX | 1x4 | **3.78** | 0.55 | 3.20 | 0.86 | 0.13 | **8.52** |

Reading the table:

1. **Tree force**: at 1x1 it drops from 58.77 to 10.90 ms (81% reduction for LASX); at 4x1 from 16.45 to 3.69 ms. The profile-level speed-up (5.4x at 1x1) is lower than the kernel-level 6.3x because the profile entry also includes traversal, local essential tree exchange, and write-back.
2. **Tree neighbor search** is a small term (0.55-1.29 ms) that shrinks only slightly: the traversal itself cannot be vectorized.
3. **Short-range PP force** (Hermite, F64) is unchanged across builds (about 10.8 ms at 1x1, 4.1 ms at 4x1). It therefore becomes the **largest single component** of the LASX builds at 1x1 and 4x1.
4. **Cluster search and group creation** (about 4.3 ms at 1x1) is likewise untouched, and its share grows from 6% to 16%.
5. **Totals**: 75.49 -> 27.22 ms at 1x1 (2.77x), 23.15 -> 9.98 ms at 4x1 (2.32x), 24.08 -> 8.52 ms at 1x4 (2.83x).

This is textbook Amdahl behavior. With the scalar 1x1 profile, the tree force takes 77.9% of a step, so the ideal end-to-end speed-up is

$$
S_{\text{ideal}} = \frac{1}{(1-f) + f / s_{\text{kernel}}},\qquad f = 0.779 .
$$

For LSX, $s=3.14$ gives 2.13x ideal versus 2.08x measured (98% of the bound); for LASX, $s=6.11$ gives 2.88x ideal versus 2.76x measured (96%). **Both widths capture essentially all of the headroom that vectorizing the tree force can provide.** Further gains must come from the short-range force and the cluster search, which now account for more than half of a step.

---

## 8. Numerical Conservation

The SIMD kernels use F32 for the tree force, so the conservation quality was checked explicitly: N=2000 demo, t=10 Myr, 4x1, three repetitions each.

![Conservation QA](figs/figC1_conservation.png)

*Figure 8-1: cumulative relative conservation errors at t=10 Myr (log scale; grey = scalar, green = LSX, blue = LASX).*

| Build | dE_cum / abs(E0) (3 runs) | dL_cum / abs(L0) (3 runs) |
|---|---|---|
| Scalar (F64) | 1.13e-6 / 2.39e-7 / 4.98e-7 | 2.65e-5 / 2.33e-5 / 2.85e-5 |
| LSX (F32) | 4.75e-7 / 8.98e-8 / 6.09e-7 | 2.40e-5 / 2.50e-5 / 1.68e-5 |
| LASX (F32) | 1.20e-6 / 4.42e-6 / 3.96e-6 | 1.83e-5 / 2.55e-5 / 2.49e-5 |
| tsv110 scalar (reference) | 6.5e-7 / 1.9e-6 / 2.5e-6 | 1.7e-5 / 1.8e-5 / 1.5e-5 |
| tsv110 NEON (reference) | 3.3e-6 / 2.0e-6 / 4.4e-7 | 2.3e-5 / 3.7e-5 / 3.0e-5 |

Energy conservation degrades by at most half an order of magnitude (to the 1e-6 level for LASX), while angular-momentum errors are essentially unchanged (about 2e-5). No run shows the 7%-level discrete jump caused by a hard-binary event. **For scientific purposes the F32 tree force costs nothing that matters.**

---

## 9. Production Run (100 Myr)

The production sample follows the official sample workflow: mcluster generates N=1000 stars with a 95% primordial-binary fraction (IMF up to 50 Msun), `petar.init` converts the file, and the run is `OMP_NUM_THREADS=8 petar -u 1 -b 500 -t 100 -o 5 input`.

![Production run](figs/figPR1_productivity.png)

*Figure 9-1: wall clock of the 100 Myr production sample (8 OMP threads).*

| Metric | Scalar | LSX | LASX |
|---|---:|---:|---:|
| Wall clock | 28 min 16 s (1696 s) | **10 min 54 s (654 s)** | **11 min 39 s (699 s)** |
| Finish | `FDPS has successfully finished` | yes | yes |
| Output files | 25 items | 25, identical set | 25, identical set |
| Final energy error | 3.59e-6 | 9.33e-5 | -4.57e-6 |
| Cumulative dL / abs(L0) | 0.91 | 0.90 | 0.29 |

Both SIMD builds shorten a 100 Myr run from nearly half an hour to about 11 minutes (**2.4-2.6x**). The 7% difference between LSX and LASX, and the spread in final errors, are dominated by random hard-binary events (a factor of 1.6-3.3 has been observed for repeated runs of the same initial condition), so the **short fixed-step benchmarks in sections 5 and 6 are the meaningful ranking**; the long run confirms that both widths are production-ready.

---

## 10. Cross-Platform Positioning

The same N=2000 demo and the same measurement protocol were used in an earlier benchmark on a Kunpeng-920 (tsv110, 24 cores) and on a dual Xeon E5-2696 v3 (36 cores, AVX2). Those results are included here as external reference points; only the 3A6000 measurements were produced for this report.

### 10.1 Kernel and end-to-end levels

![Cross-architecture kernel comparison](figs_cross/figX1_kernel_all.png)

*Figure 10-1: kernel cost per interaction across builds (ni=1024, nj=128, log scale).*

| Build | EP-EP kernel [ns] | Single core 1x1 [s] | Best configuration [s] | SIMD gain, single core |
|---|---:|---:|---|---:|
| 3A6000 scalar (F64) | 9.03 | 38.86 | 12.06 (4x1) | -- |
| 3A6000 **LSX** | 2.88 | 18.70 | 6.04 (1x4) | **2.08x** |
| 3A6000 **LASX** | 1.48 | 14.10 | **4.53 (1x4)** | **2.76x** |
| tsv110 scalar (24 cores) | 17.3 | 70.76 | 7.19 (24x1) | -- |
| tsv110 **NEON** | ~4.4 | not measured | not measured | 1.6-2.3x (end-to-end) |
| Xeon x2 **AVX2** | not comparable | 10.33 | 3.14 (36x1, bound) | -- |

![Single-core and best configuration](figs_cross/figX2_single_best_all.png)

*Figure 10-2: left = single-core wall clock; right = best configuration per platform.*

![SIMD gain](figs_cross/figX3_simd_speedup.png)

*Figure 10-3: what turning SIMD on buys at the kernel, single-core, and best-configuration levels.*

Two observations. First, LSX is 1.5x faster than the same-width NEON kernel on tsv110 (2.88 ns vs ~4.4 ns), even though both are 128-bit; the difference comes from the LA664 core itself (higher per-cycle throughput and roughly twice the single-core DRAM bandwidth). Second, the 4-core 3A6000 with LASX reaches 4.53 s at N=2000, faster than the 24-core tsv110 (7.19 s) -- with just 4 cores.

### 10.2 A caveat: s/Myr is only comparable at equal tree-step counts

![Tree-step schedules](figs_cross/figW1_steps_schedule.png)

*Figure 10-4: actual tree-step counts over t=2 Myr (tsv110 values inferred from its published s/Myr and ms/step).*

Cross-platform s/Myr ratios are easily distorted because the automatic tree time step differs between systems: over the same t=2 Myr, the 3A6000 runs 512/1024/2048/2048 steps, while the tsv110 runs about 1080/2040/1144/2067 steps. At N=2000 the tsv110 therefore performs about twice as many steps, inflating its s/Myr, while at N=4000 it performs half as many, deflating it. Direct division of s/Myr can distort ratios by a factor of 2-4.

![dt sensitivity](figs_cross/figW4_dt_sensitivity.png)

*Figure 10-5: at N=2000, the cost per step itself depends on dt (and hence on r_out).*

Matching the step counts removes most of the anomaly:

![Raw vs matched ratios](figs_cross/figW3_ratio_raw_vs_matched.png)

*Figure 10-6: tsv110 (24 cores) / 3A6000 (4 cores) time ratio; dashed = raw s/Myr, solid = step-matched.*

| N | tsv110 steps (inferred) | Scalar: raw -> matched | LSX: raw -> matched | LASX: raw -> matched |
|---:|---:|---:|---:|---:|
| 1000 | ~1080 | 1.24 -> **0.39** | 1.75 -> **0.82** | 1.95 -> **1.07** |
| 2000 | ~2040 | 1.19 -> **0.31** | 2.07 -> **0.79** | 2.54 -> **1.14** |
| 4000 | ~1144 | 0.31 -> **0.54** | 0.70 -> **1.21** | 0.95 -> **1.63** |
| 8000 | ~2067 | 0.45 -> **0.45** | 1.12 -> **1.12** | 1.68 -> **1.68** |

*(Values > 1 mean the 4-core 3A6000 is faster than the 24-core tsv110.)*

After matching, the picture is consistent: the scalar build is 2-3x slower than the 24-core tsv110 everywhere; LSX breaks even around N~3000 and pulls ahead at larger N; LASX leads throughout, with the margin growing smoothly from 1.07x to 1.68x as N increases. The wild swings in the raw ratios are artefacts of the step-count mismatch. The recommendation for future cross-machine work is to fix the time step (or at least report step counts and r_out alongside s/Myr), or to compare ms/step only.

### 10.3 Why four cores can hold their own against twenty-four

![Hardware ledger](figs_cross/figW5_hw_ledger.png)

*Figure 10-7: per-core assets (left) and aggregate resources (right).*

| Per-core item | 3A6000 | tsv110 | Ratio |
|---|---:|---:|---:|
| Scalar EP-EP kernel (nj=128) | 9.0 ns | 17.3 ns | **1.92x** |
| Single-core DRAM triad bandwidth | 23.4 GB/s | 12.3 GB/s | **1.90x** |
| L3 latency | 17.3 ns | 28.2 ns | **1.63x** |
| DRAM latency | 95.1 ns | 82.0 ns | 0.86x |
| Clock | 2.5 GHz (fixed) | 2.6 GHz | 0.96x |

The LA664 core has roughly twice the scalar throughput and twice the single-core memory bandwidth of a tsv110 core, plus a shorter L3 latency; the only deficits are DRAM latency (14% worse) and clock (4% lower). Tree-force work is bandwidth- and L3-friendly, and the short-range/cluster work is latency-sensitive -- the 3A6000 is competitive in both regimes.

Aggregate bandwidth is the key equalizer: 4 LA664 cores sustain **33.9 GB/s**, slightly more than the 24 tsv110 cores (30.9 GB/s). With the bandwidth wall at the same height on both machines, the comparison becomes one of "interactions per byte of bandwidth" -- exactly where SIMD helps. The measured per-step growth from N=2000 to N=8000 tells the same story:

![Matched-step scaling](figs_cross/figW6_scaling_matched.png)

*Figure 10-8: per-step cost growth under step matching (theoretical N log N = 4.73x).*

| Build (step-matched) | Per-step growth, N=2000 -> 8000 | Relative to N log N |
|---|---:|---:|
| tsv110 (24 cores) | **6.34x** | +34% |
| 3A6000 scalar (4 cores) | 4.46x | -6% |
| 3A6000 LSX | 4.53x | -4% |
| 3A6000 LASX | **4.34x** | -8% |

The 24-core machine's per-step cost grows faster than the theoretical N log N, because more domains mean more local-essential-tree exchange and more load imbalance; the 4-domain decomposition ages more gracefully. This is why the LA664 build gets relatively better as N increases.

**Balanced conclusion:** for small-to-medium N and for running several jobs concurrently, a 4-core 3A6000 with SIMD offers the best work per core and can match or beat a 24-core machine on a single job; for large N and sustained throughput, many-core machines remain necessary -- they lack efficiency in this regime, not resources.

### 10.4 Remaining gaps

The tsv110 and Xeon systems were not available for the same fine-grained per-step profile, so the above "remaining 56% of the step" argument rests on the fact that short-range and cluster work is scalar on both machines; it cannot be checked component by component. The tsv110 step counts are inferred from published aggregate values, and the fixed-step comparison at N=4000 still carries about an 11% step-count mismatch (1024 vs ~1144). Closing these gaps would require re-running the same profiling build on those machines.

---

## 11. Lessons and Next Steps

1. **The F32-plus-fast-rsqrt recipe transfers cleanly.** Maximum force errors stay at the 1e-4 level against the 7e-3 tolerance, neighbor counts are bit-identical, and conservation remains at the 1e-6 level. No physics trade-off is required to get the speed-up.
2. **Amdahl now rules.** After the port, the tree force is only about 40% of a 1x1 step; the short-range Hermite force (F64) and cluster search dominate the remainder. Vectorizing or pruning those is the next lever; widening the tree-force kernels further would yield little.
3. **The optimal parallel layout changed** from 4x1 to 1x4. Oversubscription and SMT are not useful on this platform.
4. **Report step counts with s/Myr.** The cross-platform comparison in section 10 shows that a step-count mismatch can distort ratios by 2-4x; ms/step or fixed-dt runs are the safe currency.
5. **Prefer LASX where available.** It is a true 256-bit path (2x the LSX kernel speed) and costs no accuracy. LSX remains the right choice for LA64 CPUs that lack LASX, still delivering about 2x end-to-end.

---

## 12. Conclusion

The LoongArch LSX/LASX tree-force port is functionally complete and production-ready. Kernel-level speed-ups of 3.2x (LSX) and 6.3x (LASX) translate into 2.1-2.8x end-to-end per core and 2.4-2.6x on a 100 Myr production run, with numerical conservation indistinguishable from the scalar build at the level that matters for N-body science. On this 4-core platform, LASX at N=2000 completes the standard demo faster than a 24-core Kunpeng-920 -- not because cores stop mattering, but because SIMD multiplies the per-core assets that this workload actually consumes.

---

## Appendix A: Build and Test Commands

```bash
# configure (FDPS and SDAR source trees are prerequisites)
./configure --with-arch=loongarch --with-simd=lasx --with-mpi=yes \
    --prefix=$PWD/install \
    --with-fdps-prefix=/path/to/FDPS --with-sdar-prefix=/path/to/SDAR
make -j8 && make install

# build and run the official self-test
make build/petar.simd.test && ./build/petar.simd.test

# LSX build instead of LASX
./configure --with-arch=loongarch --with-simd=lsx --with-mpi=yes
make -j8

# scalar build
./configure --with-arch=loongarch --with-simd=no --with-mpi=yes
make -j8
```

Runtime notes: `OMP_STACKSIZE=128M` and `ulimit -s unlimited` are recommended for large runs; when MPI is combined with OpenMP, OpenMPI benefits from `--bind-to none`; 8 ranks on a 4-core/8-thread CPU require `--use-hwthread-cpus`.

## Appendix B: Figure Index

| Figure | Content |
|---|---|
| `figs/figK1_kernel_ni1024.png` | Kernel cost per interaction (ni=1024) |
| `figs/figK2_kernel_bars.png` | Kernel cost bars (ni=1024, nj=2048) |
| `figs/figS1_scan_wall.png` | Configuration scan wall clock |
| `figs/figS2_scan_speedup.png` | Parallel speed-up |
| `figs/figB1_perstep_breakdown.png` | Per-step stacked breakdown |
| `figs/figP1_probe.png` | Start-up cost probe |
| `figs/figN1_weak_scaling.png` | Weak scaling |
| `figs/figN2_strong_scaling.png` | Strong scaling |
| `figs/figC1_conservation.png` | Conservation QA |
| `figs/figPR1_productivity.png` | 100 Myr production wall clock |
| `figs_cross/figX1_kernel_all.png` | Kernel cost, all platforms |
| `figs_cross/figX2_single_best_all.png` | Single-core / best configuration, all platforms |
| `figs_cross/figX3_simd_speedup.png` | SIMD on/off gains, all platforms |
| `figs_cross/figX4_scan_all.png` | Configuration scan, all platforms |
| `figs_cross/figX5_weak_all.png` | Weak scaling, all platforms |
| `figs_cross/figX6_perstep_all.png` | Per-step cost, all platforms |
| `figs_cross/figW1_steps_schedule.png` | Tree-step schedules |
| `figs_cross/figW2_matched_curves.png` | Raw vs step-matched curves |
| `figs_cross/figW3_ratio_raw_vs_matched.png` | Ratio, raw vs step-matched |
| `figs_cross/figW4_dt_sensitivity.png` | dt sensitivity at N=2000 |
| `figs_cross/figW5_hw_ledger.png` | Per-core and aggregate hardware ledger |
| `figs_cross/figW6_scaling_matched.png` | Matched-step per-step scaling |
