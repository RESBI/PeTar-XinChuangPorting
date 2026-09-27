#pragma once
#ifdef USE_LARCH_SIMD
// LoongArch LSX/LASX SIMD force kernels for PeTar
// ------------------------------------------------------------------
// Design notes:
//  * Tree force (EP-EP, EP-SP) is computed in F32 with LASX (8 x f32)
//    or LSX (4 x f32) vectors; positions are translated by the first
//    active i-particle to suppress cancellation.
//  * 1/sqrt() uses a bit-magic seed + 3 Newton-Raphson iterations
//    (LA664 has no frecipe estimate instruction; this is the fastest
//    accurate path found on the machine).
//  * Neighbour counting follows the scalar kernel semantics exactly:
//    the F32 count is cross-checked per pair against a relative margin
//    band; any i-particle touching the band is re-counted in F64.
//  * Two kernel shapes are provided and dispatched automatically:
//      Kernel_IV_J1: vector over i (one scalar j at a time)
//      Kernel_I1_JV: vector over j (one scalar i at a time)
// ------------------------------------------------------------------

#include <lsxintrin.h>
#include <lasxintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "soft_ptcl.hpp"

namespace larch {

using F32 = float;
using F64 = double;
using S32 = std::int32_t;
using S64 = std::int64_t;

// ------------------------------------------------------------------
// vector abstraction: LASX (256 bit) or LSX (128 bit)
// ------------------------------------------------------------------
#if defined(LARCH_SIMD_LSX)
using VF = __m128;
using VI = __m128i;
static const S32 VW = 4;
#define LARCH_VLD(p)          ((VF)__lsx_vld((p), 0))
#define LARCH_VST(v, p)       __lsx_vst((__m128i)(v), (p), 0)
#define LARCH_VADD(a, b)      __lsx_vfadd_s((a), (b))
#define LARCH_VSUB(a, b)      __lsx_vfsub_s((a), (b))
#define LARCH_VMUL(a, b)      __lsx_vfmul_s((a), (b))
#define LARCH_VMADD(a, b, c)  __lsx_vfmadd_s((a), (b), (c))
#define LARCH_VNMSUB(a, b, c) __lsx_vfnmsub_s((a), (b), (c))
#define LARCH_VMAX(a, b)      __lsx_vfmax_s((a), (b))
#define LARCH_VCMPLT(a, b)    __lsx_vfcmp_clt_s((a), (b))
#define LARCH_VAND(a, b)      __lsx_vand_v((a), (b))
#define LARCH_VOR(a, b)       __lsx_vor_v((a), (b))
#define LARCH_VADDW(a, b)     __lsx_vadd_w((a), (b))
#define LARCH_VSUBW(a, b)     __lsx_vsub_w((a), (b))
#define LARCH_VSRLIW(a, n)    __lsx_vsrli_w((a), (n))
#else
using VF = __m256;
using VI = __m256i;
static const S32 VW = 8;
#define LARCH_VLD(p)          ((VF)__lasx_xvld((p), 0))
#define LARCH_VST(v, p)       __lasx_xvst((__m256i)(v), (p), 0)
#define LARCH_VADD(a, b)      __lasx_xvfadd_s((a), (b))
#define LARCH_VSUB(a, b)      __lasx_xvfsub_s((a), (b))
#define LARCH_VMUL(a, b)      __lasx_xvfmul_s((a), (b))
#define LARCH_VMADD(a, b, c)  __lasx_xvfmadd_s((a), (b), (c))
#define LARCH_VNMSUB(a, b, c) __lasx_xvfnmsub_s((a), (b), (c))
#define LARCH_VMAX(a, b)      __lasx_xvfmax_s((a), (b))
#define LARCH_VCMPLT(a, b)    __lasx_xvfcmp_clt_s((a), (b))
#define LARCH_VAND(a, b)      __lasx_xvand_v((a), (b))
#define LARCH_VOR(a, b)       __lasx_xvor_v((a), (b))
#define LARCH_VADDW(a, b)     __lasx_xvadd_w((a), (b))
#define LARCH_VSUBW(a, b)     __lasx_xvsub_w((a), (b))
#define LARCH_VSRLIW(a, n)    __lasx_xvsrli_w((a), (n))
#endif

#if defined(LARCH_SIMD_LSX)
static inline VF vrepl(const F32 x) { return (VF)__lsx_vldrepl_w(&x, 0); }
static inline VF vrepl_mem(const F32 * p) { return (VF)__lsx_vldrepl_w(p, 0); }
static inline VI vrepli(const S32 x) { return (VI)__lsx_vldrepl_w(&x, 0); }
#else
static inline VF vrepl(const F32 x) { return (VF)__lasx_xvldrepl_w(&x, 0); }
static inline VF vrepl_mem(const F32 * p) { return (VF)__lasx_xvldrepl_w(p, 0); }
static inline VI vrepli(const S32 x) { return (VI)__lasx_xvldrepl_w(&x, 0); }
#endif

static inline VI vzero_i() { return vrepli(0); }
static inline VF vzero_f() { return vrepl(0.0f); }

static inline F32 hadd(const VF v) {
    F32 t[VW];
    LARCH_VST(v, t);
    F32 s = 0.0f;
    for (S32 k = 0; k < VW; k++) s += t[k];
    return s;
}
static inline S32 hadd_i(const VI v) {
    S32 t[VW];
    LARCH_VST((VF)v, t);
    S32 s = 0;
    for (S32 k = 0; k < VW; k++) s += t[k];
    return s;
}
static inline bool any_true(const VI v) {
    S32 t[VW];
    LARCH_VST((VF)v, t);
    for (S32 k = 0; k < VW; k++) if (t[k] != 0) return true;
    return false;
}

// 1/sqrt for F32 vectors.
// The LA664 core has no frecipe estimate instruction, so the seed comes
// from the bit-magic constant; the correction steps then follow the same
// formulas as the upstream SIMD ports:
//   step 1: x86 phantomquad RSQRT_NR_EPJ_X2 Newton step, x*(3-a*x*x)/2
//   step 2: FUGAKU force_fugaku.hpp cubic correction, x*(1 + h*(0.5+0.375h))
// Step 1 lifts the 5-bit magic seed to ~11 bit (comparable with the
// hardware rsqrt estimate used on x86/A64FX); step 2 then reaches full
// F32 accuracy. max relative error ~1e-7 measured on la664.
static inline VF rsqrt_fast(const VF a) {
    static const S32 magic = 0x5f3759df;
    VI ib = (VI)a;
    ib = LARCH_VSUBW(vrepli(magic), LARCH_VSRLIW(ib, 1));
    VF x = (VF)ib;
    const VF half = vrepl(0.5f);
    const VF one = vrepl(1.0f);
    const VF three = vrepl(3.0f);
    // x86 RSQRT_NR_EPJ_X2: x = x*(3 - a*x*x)*0.5
    VF x2 = LARCH_VMUL(x, x);
    VF h = LARCH_VNMSUB(a, x2, three);
    x = LARCH_VMUL(LARCH_VMUL(x, h), half);
    // FUGAKU cubic: h = 1 - a*x*x; x = x + x*((0.375*h + 0.5)*h)
    x2 = LARCH_VMUL(x, x);
    h = LARCH_VNMSUB(a, x2, one);
    VF poly = LARCH_VMADD(h, vrepl(0.375f), half);
    poly = LARCH_VMUL(poly, h);
    x = LARCH_VMADD(x, poly, x);
    return x;
}

// relative half-width of the r2/R2 margin band used for neighbour-count
// cross checking against the F64 reference (2^-10)
static constexpr F32 NGB_MARGIN = 1.0f / 1024.0f;
// position used to pad inactive lanes; 1e10 is far outside any realistic
// search radius while (1e10)^2 stays finite in F32 (no inf/NaN)
static constexpr F32 PAD_POS = 1e10f;

// ------------------------------------------------------------------
// per-thread scratch buffers
// A plain POD with raw pointers: thread_local objects without dynamic
// initialisation avoid the __tls_init call on every access.
// ------------------------------------------------------------------
struct LarchBuf {
    S32 cap_i, cap_j;
    F32 *ix, *iy, *iz, *irs2;
    S32 *ilist;
    F32 *jx, *jy, *jz, *jm, *jrs2;
    F32 *qxx, *qyy, *qzz, *qxy, *qxz, *qyz;

    static F32 * allocF(F32 * old, const S32 m) {
        std::free(old);
        F32 * p = (F32 *)std::malloc(sizeof(F32) * (size_t)m);
        if (p == nullptr) { std::fprintf(stderr, "larch: out of memory (F32[%d])\n", m); std::abort(); }
        return p;
    }
    static S32 * allocS(S32 * old, const S32 m) {
        std::free(old);
        S32 * p = (S32 *)std::malloc(sizeof(S32) * (size_t)m);
        if (p == nullptr) { std::fprintf(stderr, "larch: out of memory (S32[%d])\n", m); std::abort(); }
        return p;
    }
    void need_i(const S32 n) {
        const S32 m = ((n + VW - 1) / VW) * VW + 4 * VW;
        if (cap_i >= m) return;
        cap_i = m;
        ix = allocF(ix, m); iy = allocF(iy, m); iz = allocF(iz, m);
        irs2 = allocF(irs2, m); ilist = allocS(ilist, m);
    }
    void need_j(const S32 n) {
        const S32 m = ((n + VW - 1) / VW) * VW + 4 * VW;
        if (cap_j >= m) return;
        cap_j = m;
        jx = allocF(jx, m); jy = allocF(jy, m); jz = allocF(jz, m);
        jm = allocF(jm, m); jrs2 = allocF(jrs2, m);
        qxx = allocF(qxx, m); qyy = allocF(qyy, m); qzz = allocF(qzz, m);
        qxy = allocF(qxy, m); qxz = allocF(qxz, m); qyz = allocF(qyz, m);
    }
};

static thread_local LarchBuf buf;

// ------------------------------------------------------------------
// exact F64 neighbour re-count (same semantics as the NoSimd kernel)
// ------------------------------------------------------------------
static inline S32 countNeighborExact(const EPISoft & pi,
                                     const EPJSoft * epj,
                                     const S32 n_jp) {
    const F64 xi = pi.pos.x, yi = pi.pos.y, zi = pi.pos.z;
    S32 n_ngb = 0;
    for (S32 j = 0; j < n_jp; j++) {
        const F64 dx = xi - epj[j].pos.x;
        const F64 dy = yi - epj[j].pos.y;
        const F64 dz = zi - epj[j].pos.z;
        const F64 r2 = dx*dx + dy*dy + dz*dz;
        const F64 r_search = std::max(pi.r_search, epj[j].r_search);
        if (r2 < r_search*r_search) n_ngb++;
    }
    return n_ngb;
}

// store back one i-block; optional exact neighbour re-count for the lanes
// flagged in danger
static inline void writeBackIBlock(const S32 i0,
                                   const S32 n_act,
                                   const S32 * ilist,
                                   const VF acc_x, const VF acc_y, const VF acc_z,
                                   const VF pot,
                                   const VI nnb,
                                   const VI danger,
                                   const EPISoft * epi,
                                   const EPJSoft * epj,
                                   const S32 n_jp,
                                   const bool count_exact_flag,
                                   const F64 G,
                                   ForceSoft * force) {
    F32 ax[VW], ay[VW], az[VW], pt[VW];
    S32 nb[VW], dg[VW];
    LARCH_VST(acc_x, ax); LARCH_VST(acc_y, ay); LARCH_VST(acc_z, az);
    LARCH_VST(pot, pt);
    LARCH_VST((VF)nnb, nb);
    LARCH_VST((VF)danger, dg);
    for (S32 k = 0; k < VW; k++) {
        const S32 kk = i0 + k;
        if (kk >= n_act) break;
        S32 cnt = nb[k];
        if (dg[k] && count_exact_flag)
            cnt = countNeighborExact(epi[ilist[kk]], epj, n_jp);
        force[ilist[kk]].acc.x += G * (F64)ax[k];
        force[ilist[kk]].acc.y += G * (F64)ay[k];
        force[ilist[kk]].acc.z += G * (F64)az[k];
        force[ilist[kk]].pot   += G * (F64)pt[k];
        if (count_exact_flag) force[ilist[kk]].n_ngb += cnt;
    }
}

// ==================================================================
// neighbour search
// ==================================================================
struct SearchNeighborEpEpSimd {
    void operator()(const EPISoft * epi, const S32 n_ip,
                    const EPJSoft * epj, const S32 n_jp,
                    ForceSoft * force) {
        if (n_ip <= 0 || n_jp <= 0) return;
        if (n_ip < VW) Kernel_I1_JV(epi, n_ip, epj, n_jp, force);
        else           Kernel_IV_J1(epi, n_ip, epj, n_jp, force);
    }

    void Kernel_IV_J1(const EPISoft * epi, const S32 n_ip,
                      const EPJSoft * epj, const S32 n_jp,
                      ForceSoft * force) {
        const F64 x0 = epi[0].pos.x, y0 = epi[0].pos.y, z0 = epi[0].pos.z;
        buf.need_i(n_ip);
        for (S32 i = 0; i < n_ip; i++) {
            buf.ilist[i] = i;
            buf.ix[i] = (F32)(epi[i].pos.x - x0);
            buf.iy[i] = (F32)(epi[i].pos.y - y0);
            buf.iz[i] = (F32)(epi[i].pos.z - z0);
            const F32 rs = (F32)epi[i].r_search;
            buf.irs2[i] = rs * rs;
        }
        const S32 n_pad = ((n_ip + VW - 1) / VW) * VW;
        for (S32 k = n_ip; k < n_pad + VW; k++) {
            buf.ix[k] = PAD_POS; buf.iy[k] = PAD_POS; buf.iz[k] = PAD_POS;
            buf.irs2[k] = 0.0f;
        }
        buf.need_j(n_jp);
        for (S32 j = 0; j < n_jp; j++) {
            buf.jx[j] = (F32)(epj[j].pos.x - x0);
            buf.jy[j] = (F32)(epj[j].pos.y - y0);
            buf.jz[j] = (F32)(epj[j].pos.z - z0);
            const F32 rs = (F32)epj[j].r_search;
            buf.jrs2[j] = rs * rs;
        }
        for (S32 k = n_jp; k < n_jp + VW; k++) {
            buf.jx[k] = PAD_POS; buf.jy[k] = PAD_POS; buf.jz[k] = PAD_POS;
            buf.jrs2[k] = 0.0f;
        }
        const VF vmargin = vrepl(NGB_MARGIN);
        const VI vone = vrepli(1);
        for (S32 i0 = 0; i0 < n_pad; i0 += VW) {
            const VF xi = LARCH_VLD(&buf.ix[i0]);
            const VF yi = LARCH_VLD(&buf.iy[i0]);
            const VF zi = LARCH_VLD(&buf.iz[i0]);
            const VF rsi2 = LARCH_VLD(&buf.irs2[i0]);
            VI nnb = vzero_i();
            VI danger = vzero_i();
            for (S32 j = 0; j < n_jp; j++) {
                const VF dx = LARCH_VSUB(xi, vrepl_mem(&buf.jx[j]));
                const VF dy = LARCH_VSUB(yi, vrepl_mem(&buf.jy[j]));
                const VF dz = LARCH_VSUB(zi, vrepl_mem(&buf.jz[j]));
                VF r2 = LARCH_VMADD(dx, dx, vzero_f());
                r2 = LARCH_VMADD(dy, dy, r2);
                r2 = LARCH_VMADD(dz, dz, r2);
                const VF R2 = LARCH_VMAX(rsi2, vrepl_mem(&buf.jrs2[j]));
                nnb = LARCH_VADDW(nnb, LARCH_VAND(LARCH_VCMPLT(r2, R2), vone));
                const VF diff = LARCH_VSUB(r2, R2);
                const VF adiff = (VF)LARCH_VAND((VI)diff, vrepli(0x7fffffff));
                danger = LARCH_VOR(danger, LARCH_VCMPLT(adiff, LARCH_VMUL(R2, vmargin)));
            }
            writeBackIBlock(i0, n_ip, buf.ilist, vzero_f(), vzero_f(), vzero_f(),
                            vzero_f(), nnb, danger, epi, epj, n_jp, true, 0.0, force);
        }
    }

    void Kernel_I1_JV(const EPISoft * epi, const S32 n_ip,
                      const EPJSoft * epj, const S32 n_jp,
                      ForceSoft * force) {
        const F64 x0 = epi[0].pos.x, y0 = epi[0].pos.y, z0 = epi[0].pos.z;
        buf.need_j(n_jp);
        for (S32 j = 0; j < n_jp; j++) {
            buf.jx[j] = (F32)(epj[j].pos.x - x0);
            buf.jy[j] = (F32)(epj[j].pos.y - y0);
            buf.jz[j] = (F32)(epj[j].pos.z - z0);
            const F32 rs = (F32)epj[j].r_search;
            buf.jrs2[j] = rs * rs;
        }
        const S32 n_pad = ((n_jp + VW - 1) / VW) * VW;
        for (S32 j = n_jp; j < n_pad + VW; j++) {
            buf.jx[j] = PAD_POS; buf.jy[j] = PAD_POS; buf.jz[j] = PAD_POS;
            buf.jrs2[j] = 0.0f;
        }
        const VF vmargin = vrepl(NGB_MARGIN);
        for (S32 i = 0; i < n_ip; i++) {
            const VF xi = vrepl((F32)(epi[i].pos.x - x0));
            const VF yi = vrepl((F32)(epi[i].pos.y - y0));
            const VF zi = vrepl((F32)(epi[i].pos.z - z0));
            const F32 rs = (F32)epi[i].r_search;
            const VF rsi2 = vrepl(rs * rs);
            S32 nnb = 0;
            bool danger_flag = false;
            for (S32 j0 = 0; j0 < n_pad; j0 += VW) {
                const VF R2 = LARCH_VMAX(rsi2, LARCH_VLD(&buf.jrs2[j0]));
                const VF dx = LARCH_VSUB(xi, LARCH_VLD(&buf.jx[j0]));
                const VF dy = LARCH_VSUB(yi, LARCH_VLD(&buf.jy[j0]));
                const VF dz = LARCH_VSUB(zi, LARCH_VLD(&buf.jz[j0]));
                VF r2 = LARCH_VMADD(dx, dx, vzero_f());
                r2 = LARCH_VMADD(dy, dy, r2);
                r2 = LARCH_VMADD(dz, dz, r2);
                nnb += hadd_i(LARCH_VAND(LARCH_VCMPLT(r2, R2), vrepli(1)));
                const VF diff = LARCH_VSUB(r2, R2);
                const VF adiff = (VF)LARCH_VAND((VI)diff, vrepli(0x7fffffff));
                if (any_true(LARCH_VCMPLT(adiff, LARCH_VMUL(R2, vmargin))))
                    danger_flag = true;
            }
            S32 cnt = nnb;
            if (danger_flag) cnt = countNeighborExact(epi[i], epj, n_jp);
            force[i].n_ngb += cnt;
        }
    }
};

// ==================================================================
// EP-EP tree force
// ==================================================================
struct CalcForceEpEpWithLinearCutoffSimd {
    void operator()(const EPISoft * epi, const S32 n_ip,
                    const EPJSoft * epj, const S32 n_jp,
                    ForceSoft * force) {
        if (n_ip <= 0 || n_jp <= 0) return;
        S32 n_act = 0;
        for (S32 i = 0; i < n_ip; i++) if (epi[i].type == 1) n_act++;
        if (n_act < VW) Kernel_I1_JV(epi, n_ip, epj, n_jp, force);
        else            Kernel_IV_J1(epi, n_ip, epj, n_jp, force);
    }

    void Kernel_IV_J1(const EPISoft * epi, const S32 n_ip,
                      const EPJSoft * epj, const S32 n_jp,
                      ForceSoft * force) {
        const F64 eps2 = EPISoft::eps * EPISoft::eps;
        const F64 r_out2 = EPISoft::r_out * EPISoft::r_out;
        const F64 G = ForceSoft::grav_const;
        F64 x0 = 0.0, y0 = 0.0, z0 = 0.0;
        bool have_origin = false;
        buf.need_i(n_ip);
        S32 n_act = 0;
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type != 1) continue;
            if (!have_origin) {
                x0 = epi[i].pos.x; y0 = epi[i].pos.y; z0 = epi[i].pos.z;
                have_origin = true;
            }
            buf.ilist[n_act] = i;
            buf.ix[n_act] = (F32)(epi[i].pos.x - x0);
            buf.iy[n_act] = (F32)(epi[i].pos.y - y0);
            buf.iz[n_act] = (F32)(epi[i].pos.z - z0);
            const F32 rs = (F32)epi[i].r_search;
            buf.irs2[n_act] = rs * rs;
            n_act++;
        }
        if (n_act == 0) return;
        const S32 n_pad = ((n_act + VW - 1) / VW) * VW;
        for (S32 k = n_act; k < n_pad + VW; k++) {
            buf.ix[k] = PAD_POS; buf.iy[k] = PAD_POS; buf.iz[k] = PAD_POS;
            buf.irs2[k] = 0.0f;
        }
        buf.need_j(n_jp);
        S32 nj_act = 0;
        for (S32 j = 0; j < n_jp; j++) {
            buf.jx[nj_act] = (F32)(epj[j].pos.x - x0);
            buf.jy[nj_act] = (F32)(epj[j].pos.y - y0);
            buf.jz[nj_act] = (F32)(epj[j].pos.z - z0);
            buf.jm[nj_act] = (F32)epj[j].mass;
            const F32 rs = (F32)epj[j].r_search;
            buf.jrs2[nj_act] = rs * rs;
            nj_act++;
        }
        for (S32 k = nj_act; k < nj_act + VW; k++) {
            buf.jx[k] = PAD_POS; buf.jy[k] = PAD_POS; buf.jz[k] = PAD_POS;
            buf.jm[k] = 0.0f; buf.jrs2[k] = 0.0f;
        }
        const VF veps2 = vrepl((F32)eps2);
        const VF vrcut2 = vrepl((F32)r_out2);
        const VF vmargin = vrepl(NGB_MARGIN);
        const VI vone = vrepli(1);
        for (S32 i0 = 0; i0 < n_pad; i0 += VW) {
            const VF xi = LARCH_VLD(&buf.ix[i0]);
            const VF yi = LARCH_VLD(&buf.iy[i0]);
            const VF zi = LARCH_VLD(&buf.iz[i0]);
            const VF rsi2 = LARCH_VLD(&buf.irs2[i0]);
            VF ax = vzero_f(), ay = vzero_f(), az = vzero_f(), pt = vzero_f();
            VI nnb = vzero_i();
            VI danger = vzero_i();
            for (S32 j = 0; j < nj_act; j++) {
                const VF dx = LARCH_VSUB(xi, vrepl_mem(&buf.jx[j]));
                const VF dy = LARCH_VSUB(yi, vrepl_mem(&buf.jy[j]));
                const VF dz = LARCH_VSUB(zi, vrepl_mem(&buf.jz[j]));
                const VF mj = vrepl_mem(&buf.jm[j]);
                VF r2 = LARCH_VMADD(dx, dx, vzero_f());
                r2 = LARCH_VMADD(dy, dy, r2);
                r2 = LARCH_VMADD(dz, dz, r2);
                const VF R2 = LARCH_VMAX(rsi2, vrepl_mem(&buf.jrs2[j]));
                nnb = LARCH_VADDW(nnb, LARCH_VAND(LARCH_VCMPLT(r2, R2), vone));
                const VF diff = LARCH_VSUB(r2, R2);
                const VF adiff = (VF)LARCH_VAND((VI)diff, vrepli(0x7fffffff));
                danger = LARCH_VOR(danger, LARCH_VCMPLT(adiff, LARCH_VMUL(R2, vmargin)));
                const VF r2_eps = LARCH_VADD(r2, veps2);
                const VF r2_cut = LARCH_VMAX(r2_eps, vrcut2);
                const VF r_inv = rsqrt_fast(r2_cut);
                const VF r2_inv = LARCH_VMUL(r_inv, r_inv);
                const VF mr = LARCH_VMUL(mj, r_inv);
                const VF mr3 = LARCH_VMUL(mr, r2_inv);
                ax = LARCH_VNMSUB(mr3, dx, ax);
                ay = LARCH_VNMSUB(mr3, dy, ay);
                az = LARCH_VNMSUB(mr3, dz, az);
                pt = LARCH_VSUB(pt, mr);
            }
            writeBackIBlock(i0, n_act, buf.ilist, ax, ay, az, pt, nnb, danger,
                            epi, epj, n_jp, true, G, force);
        }
    }

    void Kernel_I1_JV(const EPISoft * epi, const S32 n_ip,
                      const EPJSoft * epj, const S32 n_jp,
                      ForceSoft * force) {
        const F64 eps2 = EPISoft::eps * EPISoft::eps;
        const F64 r_out2 = EPISoft::r_out * EPISoft::r_out;
        const F64 G = ForceSoft::grav_const;
        F64 x0 = 0.0, y0 = 0.0, z0 = 0.0;
        bool have_origin = false;
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type == 1) {
                x0 = epi[i].pos.x; y0 = epi[i].pos.y; z0 = epi[i].pos.z;
                have_origin = true; break;
            }
        }
        if (!have_origin) return;
        buf.need_j(n_jp);
        S32 nj_act = 0;
        for (S32 j = 0; j < n_jp; j++) {
            buf.jx[nj_act] = (F32)(epj[j].pos.x - x0);
            buf.jy[nj_act] = (F32)(epj[j].pos.y - y0);
            buf.jz[nj_act] = (F32)(epj[j].pos.z - z0);
            buf.jm[nj_act] = (F32)epj[j].mass;
            const F32 rs = (F32)epj[j].r_search;
            buf.jrs2[nj_act] = rs * rs;
            nj_act++;
        }
        const S32 n_pad = ((nj_act + VW - 1) / VW) * VW;
        for (S32 k = nj_act; k < n_pad + VW; k++) {
            buf.jx[k] = PAD_POS; buf.jy[k] = PAD_POS; buf.jz[k] = PAD_POS;
            buf.jm[k] = 0.0f; buf.jrs2[k] = 0.0f;
        }
        const VF veps2 = vrepl((F32)eps2);
        const VF vrcut2 = vrepl((F32)r_out2);
        const VF vmargin = vrepl(NGB_MARGIN);
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type != 1) continue;
            const VF xi = vrepl((F32)(epi[i].pos.x - x0));
            const VF yi = vrepl((F32)(epi[i].pos.y - y0));
            const VF zi = vrepl((F32)(epi[i].pos.z - z0));
            const F32 rs = (F32)epi[i].r_search;
            const VF rsi2 = vrepl(rs * rs);
            VF ax = vzero_f(), ay = vzero_f(), az = vzero_f(), pt = vzero_f();
            S32 nnb = 0;
            bool danger_flag = false;
            for (S32 j0 = 0; j0 < n_pad; j0 += VW) {
                const VF mj = LARCH_VLD(&buf.jm[j0]);
                const VF R2 = LARCH_VMAX(rsi2, LARCH_VLD(&buf.jrs2[j0]));
                const VF dx = LARCH_VSUB(xi, LARCH_VLD(&buf.jx[j0]));
                const VF dy = LARCH_VSUB(yi, LARCH_VLD(&buf.jy[j0]));
                const VF dz = LARCH_VSUB(zi, LARCH_VLD(&buf.jz[j0]));
                VF r2 = LARCH_VMADD(dx, dx, vzero_f());
                r2 = LARCH_VMADD(dy, dy, r2);
                r2 = LARCH_VMADD(dz, dz, r2);
                nnb += hadd_i(LARCH_VAND(LARCH_VCMPLT(r2, R2), vrepli(1)));
                const VF diff = LARCH_VSUB(r2, R2);
                const VF adiff = (VF)LARCH_VAND((VI)diff, vrepli(0x7fffffff));
                if (any_true(LARCH_VCMPLT(adiff, LARCH_VMUL(R2, vmargin))))
                    danger_flag = true;
                const VF r2_eps = LARCH_VADD(r2, veps2);
                const VF r2_cut = LARCH_VMAX(r2_eps, vrcut2);
                const VF r_inv = rsqrt_fast(r2_cut);
                const VF r2_inv = LARCH_VMUL(r_inv, r_inv);
                const VF mr = LARCH_VMUL(mj, r_inv);
                const VF mr3 = LARCH_VMUL(mr, r2_inv);
                ax = LARCH_VNMSUB(mr3, dx, ax);
                ay = LARCH_VNMSUB(mr3, dy, ay);
                az = LARCH_VNMSUB(mr3, dz, az);
                pt = LARCH_VSUB(pt, mr);
            }
            S32 cnt = nnb;
            if (danger_flag) cnt = countNeighborExact(epi[i], epj, n_jp);
            force[i].acc.x += G * (F64)hadd(ax);
            force[i].acc.y += G * (F64)hadd(ay);
            force[i].acc.z += G * (F64)hadd(az);
            force[i].pot   += G * (F64)hadd(pt);
            force[i].n_ngb += cnt;
        }
    }
};

// ==================================================================
// EP-SP monopole
// ==================================================================
struct CalcForceEpSpMonoSimd {
    template <class Tsp>
    void operator()(const EPISoft * epi, const S32 n_ip,
                    const Tsp * spj, const S32 n_jp,
                    ForceSoft * force) {
        if (n_ip <= 0 || n_jp <= 0) return;
        S32 n_act = 0;
        for (S32 i = 0; i < n_ip; i++) if (epi[i].type == 1) n_act++;
        if (n_act < VW) Kernel_I1_JV(epi, n_ip, spj, n_jp, force);
        else            Kernel_IV_J1(epi, n_ip, spj, n_jp, force);
    }

    template <class Tsp>
    static void loadSpj(const Tsp & sp, const F64 x0, const F64 y0, const F64 z0, const S32 j) {
        const PS::F64vec p = sp.getPos();
        buf.jx[j] = (F32)(p.x - x0);
        buf.jy[j] = (F32)(p.y - y0);
        buf.jz[j] = (F32)(p.z - z0);
        buf.jm[j] = (F32)sp.getCharge();
    }

    template <class Tsp>
    void Kernel_IV_J1(const EPISoft * epi, const S32 n_ip,
                      const Tsp * spj, const S32 n_jp,
                      ForceSoft * force) {
        const F64 eps2 = EPISoft::eps * EPISoft::eps;
        const F64 G = ForceSoft::grav_const;
        F64 x0 = 0.0, y0 = 0.0, z0 = 0.0;
        bool have_origin = false;
        buf.need_i(n_ip);
        S32 n_act = 0;
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type != 1) continue;
            if (!have_origin) { x0 = epi[i].pos.x; y0 = epi[i].pos.y; z0 = epi[i].pos.z; have_origin = true; }
            buf.ilist[n_act] = i;
            buf.ix[n_act] = (F32)(epi[i].pos.x - x0);
            buf.iy[n_act] = (F32)(epi[i].pos.y - y0);
            buf.iz[n_act] = (F32)(epi[i].pos.z - z0);
            n_act++;
        }
        if (n_act == 0) return;
        const S32 n_pad = ((n_act + VW - 1) / VW) * VW;
        for (S32 k = n_act; k < n_pad + VW; k++) {
            buf.ix[k] = PAD_POS; buf.iy[k] = PAD_POS; buf.iz[k] = PAD_POS;
        }
        buf.need_j(n_jp);
        for (S32 j = 0; j < n_jp; j++) loadSpj(spj[j], x0, y0, z0, j);
        for (S32 k = n_jp; k < n_jp + VW; k++) {
            buf.jx[k] = PAD_POS; buf.jy[k] = PAD_POS; buf.jz[k] = PAD_POS;
            buf.jm[k] = 0.0f;
        }
        const VF veps2 = vrepl((F32)eps2);
        for (S32 i0 = 0; i0 < n_pad; i0 += VW) {
            const VF xi = LARCH_VLD(&buf.ix[i0]);
            const VF yi = LARCH_VLD(&buf.iy[i0]);
            const VF zi = LARCH_VLD(&buf.iz[i0]);
            VF ax = vzero_f(), ay = vzero_f(), az = vzero_f(), pt = vzero_f();
            for (S32 j = 0; j < n_jp; j++) {
                const VF dx = LARCH_VSUB(xi, vrepl_mem(&buf.jx[j]));
                const VF dy = LARCH_VSUB(yi, vrepl_mem(&buf.jy[j]));
                const VF dz = LARCH_VSUB(zi, vrepl_mem(&buf.jz[j]));
                const VF mj = vrepl_mem(&buf.jm[j]);
                VF r2 = LARCH_VMADD(dx, dx, veps2);
                r2 = LARCH_VMADD(dy, dy, r2);
                r2 = LARCH_VMADD(dz, dz, r2);
                const VF r_inv = rsqrt_fast(r2);
                const VF mr = LARCH_VMUL(mj, r_inv);
                const VF mr3 = LARCH_VMUL(mr, LARCH_VMUL(r_inv, r_inv));
                ax = LARCH_VNMSUB(mr3, dx, ax);
                ay = LARCH_VNMSUB(mr3, dy, ay);
                az = LARCH_VNMSUB(mr3, dz, az);
                pt = LARCH_VSUB(pt, mr);
            }
            writeBackIBlock(i0, n_act, buf.ilist, ax, ay, az, pt,
                            vzero_i(), vzero_i(), epi, nullptr, 0, false, G, force);
        }
    }

    template <class Tsp>
    void Kernel_I1_JV(const EPISoft * epi, const S32 n_ip,
                      const Tsp * spj, const S32 n_jp,
                      ForceSoft * force) {
        const F64 eps2 = EPISoft::eps * EPISoft::eps;
        const F64 G = ForceSoft::grav_const;
        F64 x0 = 0.0, y0 = 0.0, z0 = 0.0;
        bool have_origin = false;
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type == 1) { x0 = epi[i].pos.x; y0 = epi[i].pos.y; z0 = epi[i].pos.z; have_origin = true; break; }
        }
        if (!have_origin) return;
        buf.need_j(n_jp);
        for (S32 j = 0; j < n_jp; j++) loadSpj(spj[j], x0, y0, z0, j);
        const S32 n_pad = ((n_jp + VW - 1) / VW) * VW;
        for (S32 k = n_jp; k < n_pad + VW; k++) {
            buf.jx[k] = PAD_POS; buf.jy[k] = PAD_POS; buf.jz[k] = PAD_POS;
            buf.jm[k] = 0.0f;
        }
        const VF veps2 = vrepl((F32)eps2);
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type != 1) continue;
            const VF xi = vrepl((F32)(epi[i].pos.x - x0));
            const VF yi = vrepl((F32)(epi[i].pos.y - y0));
            const VF zi = vrepl((F32)(epi[i].pos.z - z0));
            VF ax = vzero_f(), ay = vzero_f(), az = vzero_f(), pt = vzero_f();
            for (S32 j0 = 0; j0 < n_pad; j0 += VW) {
                const VF dx = LARCH_VSUB(xi, LARCH_VLD(&buf.jx[j0]));
                const VF dy = LARCH_VSUB(yi, LARCH_VLD(&buf.jy[j0]));
                const VF dz = LARCH_VSUB(zi, LARCH_VLD(&buf.jz[j0]));
                const VF mj = LARCH_VLD(&buf.jm[j0]);
                VF r2 = LARCH_VMADD(dx, dx, veps2);
                r2 = LARCH_VMADD(dy, dy, r2);
                r2 = LARCH_VMADD(dz, dz, r2);
                const VF r_inv = rsqrt_fast(r2);
                const VF mr = LARCH_VMUL(mj, r_inv);
                const VF mr3 = LARCH_VMUL(mr, LARCH_VMUL(r_inv, r_inv));
                ax = LARCH_VNMSUB(mr3, dx, ax);
                ay = LARCH_VNMSUB(mr3, dy, ay);
                az = LARCH_VNMSUB(mr3, dz, az);
                pt = LARCH_VSUB(pt, mr);
            }
            force[i].acc.x += G * (F64)hadd(ax);
            force[i].acc.y += G * (F64)hadd(ay);
            force[i].acc.z += G * (F64)hadd(az);
            force[i].pot   += G * (F64)hadd(pt);
        }
    }
};

// ==================================================================
// EP-SP quadrupole
// ==================================================================
struct CalcForceEpSpQuadSimd {
    template <class Tsp>
    void operator()(const EPISoft * epi, const S32 n_ip,
                    const Tsp * spj, const S32 n_jp,
                    ForceSoft * force) {
        if (n_ip <= 0 || n_jp <= 0) return;
        S32 n_act = 0;
        for (S32 i = 0; i < n_ip; i++) if (epi[i].type == 1) n_act++;
        if (n_act < VW) Kernel_I1_JV(epi, n_ip, spj, n_jp, force);
        else            Kernel_IV_J1(epi, n_ip, spj, n_jp, force);
    }

    template <class Tsp>
    static void loadSpj(const Tsp & sp, const F64 x0, const F64 y0, const F64 z0, const S32 j) {
        const PS::F64vec p = sp.getPos();
        buf.jx[j] = (F32)(p.x - x0);
        buf.jy[j] = (F32)(p.y - y0);
        buf.jz[j] = (F32)(p.z - z0);
        buf.jm[j] = (F32)sp.getCharge();
        buf.qxx[j] = (F32)sp.quad.xx;
        buf.qyy[j] = (F32)sp.quad.yy;
        buf.qzz[j] = (F32)sp.quad.zz;
        buf.qxy[j] = (F32)sp.quad.xy;
        buf.qxz[j] = (F32)sp.quad.xz;
        buf.qyz[j] = (F32)sp.quad.yz;
    }

    template <class Tsp>
    void Kernel_IV_J1(const EPISoft * epi, const S32 n_ip,
                      const Tsp * spj, const S32 n_jp,
                      ForceSoft * force) {
        const F64 eps2 = EPISoft::eps * EPISoft::eps;
        const F64 G = ForceSoft::grav_const;
        F64 x0 = 0.0, y0 = 0.0, z0 = 0.0;
        bool have_origin = false;
        buf.need_i(n_ip);
        S32 n_act = 0;
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type != 1) continue;
            if (!have_origin) { x0 = epi[i].pos.x; y0 = epi[i].pos.y; z0 = epi[i].pos.z; have_origin = true; }
            buf.ilist[n_act] = i;
            buf.ix[n_act] = (F32)(epi[i].pos.x - x0);
            buf.iy[n_act] = (F32)(epi[i].pos.y - y0);
            buf.iz[n_act] = (F32)(epi[i].pos.z - z0);
            n_act++;
        }
        if (n_act == 0) return;
        const S32 n_pad = ((n_act + VW - 1) / VW) * VW;
        for (S32 k = n_act; k < n_pad + VW; k++) {
            buf.ix[k] = PAD_POS; buf.iy[k] = PAD_POS; buf.iz[k] = PAD_POS;
        }
        buf.need_j(n_jp);
        for (S32 j = 0; j < n_jp; j++) loadSpj(spj[j], x0, y0, z0, j);
        for (S32 k = n_jp; k < n_jp + VW; k++) {
            buf.jx[k] = PAD_POS; buf.jy[k] = PAD_POS; buf.jz[k] = PAD_POS;
            buf.jm[k] = 0.0f;
            buf.qxx[k] = buf.qyy[k] = buf.qzz[k] = 0.0f;
            buf.qxy[k] = buf.qxz[k] = buf.qyz[k] = 0.0f;
        }
        const VF veps2 = vrepl((F32)eps2);
        const VF vhalf = vrepl(0.5f);
        const VF vfive = vrepl(5.0f);
        const VF vone5 = vrepl(1.5f);
        const VF vneg2 = vrepl(-2.0f);
        for (S32 i0 = 0; i0 < n_pad; i0 += VW) {
            const VF xi = LARCH_VLD(&buf.ix[i0]);
            const VF yi = LARCH_VLD(&buf.iy[i0]);
            const VF zi = LARCH_VLD(&buf.iz[i0]);
            VF ax = vzero_f(), ay = vzero_f(), az = vzero_f(), pt = vzero_f();
            for (S32 j = 0; j < n_jp; j++) {
                const VF dx = LARCH_VSUB(xi, vrepl_mem(&buf.jx[j]));
                const VF dy = LARCH_VSUB(yi, vrepl_mem(&buf.jy[j]));
                const VF dz = LARCH_VSUB(zi, vrepl_mem(&buf.jz[j]));
                const VF mj = vrepl_mem(&buf.jm[j]);
                VF r2 = LARCH_VMADD(dx, dx, veps2);
                r2 = LARCH_VMADD(dy, dy, r2);
                r2 = LARCH_VMADD(dz, dz, r2);
                const VF r_inv = rsqrt_fast(r2);
                const VF r2_inv = LARCH_VMUL(r_inv, r_inv);
                const VF r3_inv = LARCH_VMUL(r2_inv, r_inv);
                const VF r5_inv = LARCH_VMUL(LARCH_VMUL(r2_inv, r3_inv), vone5);
                const VF qxx = vrepl_mem(&buf.qxx[j]), qyy = vrepl_mem(&buf.qyy[j]), qzz = vrepl_mem(&buf.qzz[j]);
                const VF qxy = vrepl_mem(&buf.qxy[j]), qxz = vrepl_mem(&buf.qxz[j]), qyz = vrepl_mem(&buf.qyz[j]);
                const VF tr = LARCH_VADD(LARCH_VADD(qxx, qyy), qzz);
                const VF qrx = LARCH_VADD(LARCH_VMUL(qxx, dx), LARCH_VADD(LARCH_VMUL(qxy, dy), LARCH_VMUL(qxz, dz)));
                const VF qry = LARCH_VADD(LARCH_VMUL(qyy, dy), LARCH_VADD(LARCH_VMUL(qyz, dz), LARCH_VMUL(qxy, dx)));
                const VF qrz = LARCH_VADD(LARCH_VMUL(qzz, dz), LARCH_VADD(LARCH_VMUL(qxz, dx), LARCH_VMUL(qyz, dy)));
                const VF qrr = LARCH_VADD(LARCH_VMUL(qrx, dx), LARCH_VADD(LARCH_VMUL(qry, dy), LARCH_VMUL(qrz, dz)));
                const VF qrr_r5 = LARCH_VMUL(r5_inv, qrr);
                const VF qrr_r7 = LARCH_VMUL(r2_inv, qrr_r5);
                const VF A = LARCH_VADD(LARCH_VSUB(LARCH_VMUL(mj, r3_inv), LARCH_VMUL(tr, r5_inv)),
                                        LARCH_VMUL(vfive, qrr_r7));
                ax = LARCH_VNMSUB(A, dx, ax);
                ay = LARCH_VNMSUB(A, dy, ay);
                az = LARCH_VNMSUB(A, dz, az);
                ax = LARCH_VNMSUB(vneg2, LARCH_VMUL(r5_inv, qrx), ax);
                ay = LARCH_VNMSUB(vneg2, LARCH_VMUL(r5_inv, qry), ay);
                az = LARCH_VNMSUB(vneg2, LARCH_VMUL(r5_inv, qrz), az);
                VF ptmp = LARCH_VMUL(mj, r_inv);
                ptmp = LARCH_VSUB(ptmp, LARCH_VMUL(vhalf, LARCH_VMUL(tr, r3_inv)));
                pt = LARCH_VSUB(pt, LARCH_VADD(ptmp, qrr_r5));
            }
            writeBackIBlock(i0, n_act, buf.ilist, ax, ay, az, pt,
                            vzero_i(), vzero_i(), epi, nullptr, 0, false, G, force);
        }
    }

    template <class Tsp>
    void Kernel_I1_JV(const EPISoft * epi, const S32 n_ip,
                      const Tsp * spj, const S32 n_jp,
                      ForceSoft * force) {
        const F64 eps2 = EPISoft::eps * EPISoft::eps;
        const F64 G = ForceSoft::grav_const;
        F64 x0 = 0.0, y0 = 0.0, z0 = 0.0;
        bool have_origin = false;
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type == 1) { x0 = epi[i].pos.x; y0 = epi[i].pos.y; z0 = epi[i].pos.z; have_origin = true; break; }
        }
        if (!have_origin) return;
        buf.need_j(n_jp);
        for (S32 j = 0; j < n_jp; j++) loadSpj(spj[j], x0, y0, z0, j);
        const S32 n_pad = ((n_jp + VW - 1) / VW) * VW;
        for (S32 k = n_jp; k < n_pad + VW; k++) {
            buf.jx[k] = PAD_POS; buf.jy[k] = PAD_POS; buf.jz[k] = PAD_POS;
            buf.jm[k] = 0.0f;
            buf.qxx[k] = buf.qyy[k] = buf.qzz[k] = 0.0f;
            buf.qxy[k] = buf.qxz[k] = buf.qyz[k] = 0.0f;
        }
        const VF veps2 = vrepl((F32)eps2);
        const VF vhalf = vrepl(0.5f);
        const VF vfive = vrepl(5.0f);
        const VF vone5 = vrepl(1.5f);
        const VF vneg2 = vrepl(-2.0f);
        for (S32 i = 0; i < n_ip; i++) {
            if (epi[i].type != 1) continue;
            const VF xi = vrepl((F32)(epi[i].pos.x - x0));
            const VF yi = vrepl((F32)(epi[i].pos.y - y0));
            const VF zi = vrepl((F32)(epi[i].pos.z - z0));
            VF ax = vzero_f(), ay = vzero_f(), az = vzero_f(), pt = vzero_f();
            for (S32 j0 = 0; j0 < n_pad; j0 += VW) {
                const VF dx = LARCH_VSUB(xi, LARCH_VLD(&buf.jx[j0]));
                const VF dy = LARCH_VSUB(yi, LARCH_VLD(&buf.jy[j0]));
                const VF dz = LARCH_VSUB(zi, LARCH_VLD(&buf.jz[j0]));
                const VF mj = LARCH_VLD(&buf.jm[j0]);
                VF r2 = LARCH_VMADD(dx, dx, veps2);
                r2 = LARCH_VMADD(dy, dy, r2);
                r2 = LARCH_VMADD(dz, dz, r2);
                const VF r_inv = rsqrt_fast(r2);
                const VF r2_inv = LARCH_VMUL(r_inv, r_inv);
                const VF r3_inv = LARCH_VMUL(r2_inv, r_inv);
                const VF r5_inv = LARCH_VMUL(LARCH_VMUL(r2_inv, r3_inv), vone5);
                const VF qxx = LARCH_VLD(&buf.qxx[j0]), qyy = LARCH_VLD(&buf.qyy[j0]), qzz = LARCH_VLD(&buf.qzz[j0]);
                const VF qxy = LARCH_VLD(&buf.qxy[j0]), qxz = LARCH_VLD(&buf.qxz[j0]), qyz = LARCH_VLD(&buf.qyz[j0]);
                const VF tr = LARCH_VADD(LARCH_VADD(qxx, qyy), qzz);
                const VF qrx = LARCH_VADD(LARCH_VMUL(qxx, dx), LARCH_VADD(LARCH_VMUL(qxy, dy), LARCH_VMUL(qxz, dz)));
                const VF qry = LARCH_VADD(LARCH_VMUL(qyy, dy), LARCH_VADD(LARCH_VMUL(qyz, dz), LARCH_VMUL(qxy, dx)));
                const VF qrz = LARCH_VADD(LARCH_VMUL(qzz, dz), LARCH_VADD(LARCH_VMUL(qxz, dx), LARCH_VMUL(qyz, dy)));
                const VF qrr = LARCH_VADD(LARCH_VMUL(qrx, dx), LARCH_VADD(LARCH_VMUL(qry, dy), LARCH_VMUL(qrz, dz)));
                const VF qrr_r5 = LARCH_VMUL(r5_inv, qrr);
                const VF qrr_r7 = LARCH_VMUL(r2_inv, qrr_r5);
                const VF A = LARCH_VADD(LARCH_VSUB(LARCH_VMUL(mj, r3_inv), LARCH_VMUL(tr, r5_inv)),
                                        LARCH_VMUL(vfive, qrr_r7));
                ax = LARCH_VNMSUB(A, dx, ax);
                ay = LARCH_VNMSUB(A, dy, ay);
                az = LARCH_VNMSUB(A, dz, az);
                ax = LARCH_VNMSUB(vneg2, LARCH_VMUL(r5_inv, qrx), ax);
                ay = LARCH_VNMSUB(vneg2, LARCH_VMUL(r5_inv, qry), ay);
                az = LARCH_VNMSUB(vneg2, LARCH_VMUL(r5_inv, qrz), az);
                VF ptmp = LARCH_VMUL(mj, r_inv);
                ptmp = LARCH_VSUB(ptmp, LARCH_VMUL(vhalf, LARCH_VMUL(tr, r3_inv)));
                pt = LARCH_VSUB(pt, LARCH_VADD(ptmp, qrr_r5));
            }
            force[i].acc.x += G * (F64)hadd(ax);
            force[i].acc.y += G * (F64)hadd(ay);
            force[i].acc.z += G * (F64)hadd(az);
            force[i].pot   += G * (F64)hadd(pt);
        }
    }
};

} // namespace larch

// expose the kernels with the same names as the x86 SIMD classes
using larch::SearchNeighborEpEpSimd;
using larch::CalcForceEpEpWithLinearCutoffSimd;
using larch::CalcForceEpSpMonoSimd;
using larch::CalcForceEpSpQuadSimd;

#endif // USE_LARCH_SIMD
