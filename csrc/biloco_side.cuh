// BiLoCo side-CTA code, included inside namespace biloco (biloco_gemm.cu): small device helpers
// (bsvd_cl), the nibble lookup tables, and the Mode 4 side-CTA z worker (bsvd_zworker).

namespace bsvd_cl {
CUTLASS_DEVICE uint32_t smem_u32(void const* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
CUTLASS_DEVICE int64_t gtimer() { int64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
template <int Off>
CUTLASS_DEVICE float lds_f32(uint32_t addr) {
    float v;
    asm("ld.shared.f32 %0, [%1+%2];" : "=f"(v) : "r"(addr), "n"(Off));
    return v;
}
CUTLASS_DEVICE float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
CUTLASS_DEVICE float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
CUTLASS_DEVICE void lut16(float v0, float v1, float v2, float v3, float (&e)[16]) {
    const float a = v0 + v1, b = v0 - v1, c = v2 + v3, d = v2 - v3;
    const float lo[4] = {-a, b, -b, a};
    const float hi[4] = {-c, d, -d, c};
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < 16; ++p) e[p] = lo[p & 3] + hi[p >> 2];
}
CUTLASS_DEVICE uint4 ldg_v4_early(void const* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
CUTLASS_DEVICE uint32_t ldg_u32_early(void const* p) {
    uint32_t v;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
CUTLASS_DEVICE float ldg_bf16_early(void const* p) {
    unsigned short v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}
}  // namespace bsvd_cl

// Nibble LUTs: for values a0..a3 the 16 entries hold sum_i s_i a_i over the sign patterns s (bit set = +1).
template <int MaxRows>
struct NibbleLut {
    static constexpr int kMaxRows = MaxRows;
    static constexpr int kMq = (MaxRows + 3) / 4;                 // row quads (M > 1 path)
    static constexpr int kMs = MaxRows == 1 ? 1 : 20;             // floats per (nibble, pattern) entry
    static constexpr int kJs = MaxRows == 1 ? 20 : 16 * 20 + 4;   // floats per nibble (padded: bank spread)
    static constexpr int kThreads = 256;                          // side CTA threads

    // Build a nibble LUT over "values" (rows x groups of 4) into lut: entry (j, p, m) at
    // lut[j * kJs + p * kMs + m]. val4(m, j) returns values 4j..4j+3 of row m.
    template <class ValFn>
    CUTLASS_DEVICE static void
    build_lut(float* lut, int nj, int rows, ValFn const& val4, int tid) {
        if constexpr (MaxRows == 1) {
            constexpr int kMaxTasks = 8;                     // nj <= 8 * 256 (host-checked)
            float4 vv[kMaxTasks];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kMaxTasks; ++k) {
                const int j = tid + k * kThreads;
                vv[k] = (j < nj) ? val4(0, j) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kMaxTasks; ++k) {
                const int j = tid + k * kThreads;
                if (j >= nj) break;
                float e[16];
                const float4 v = vv[k];
                bsvd_cl::lut16(v.x, v.y, v.z, v.w, e);
                float4* L = reinterpret_cast<float4*>(lut + j * kJs);   // 80-byte rows: no bank rotation needed
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) L[i] = make_float4(e[4 * i], e[4 * i + 1], e[4 * i + 2], e[4 * i + 3]);
            }
        } else {
            const int mq_n = (rows + 3) / 4;
            for (int t2 = tid; t2 < nj * mq_n; t2 += kThreads) {
                const int j = t2 / mq_n;
                const int mq = t2 - j * mq_n;
                float e[4][16];
                CUTLASS_PRAGMA_UNROLL
                for (int b = 0; b < 4; ++b) {
                    const int m = 4 * mq + b;
                    if (m < rows) {
                        const float4 v = val4(m, j);
                        bsvd_cl::lut16(v.x, v.y, v.z, v.w, e[b]);
                    } else {
                        CUTLASS_PRAGMA_UNROLL
                        for (int pp = 0; pp < 16; ++pp) e[b][pp] = 0.f;
                    }
                }
                float* L = lut + j * kJs + 4 * mq;
                CUTLASS_PRAGMA_UNROLL
                for (int pp = 0; pp < 16; ++pp) {
                    *reinterpret_cast<float4*>(L + pp * kMs) = make_float4(e[0][pp], e[1][pp], e[2][pp], e[3][pp]);
                }
            }
        }
    }

    // M == 1, exactly kTasks * kThreads nibbles: no guards (each guard is a branch, ~20 cycles on this part)
    template <int kTasks, class ValFn>
    CUTLASS_DEVICE static void
    build_lut_exact(float* lut, ValFn const& val4, int tid) {
        float4 vv[kTasks];
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kTasks; ++k) vv[k] = val4(0, tid + k * kThreads);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kTasks; ++k) {
            float e[16];
            const float4 v = vv[k];
            bsvd_cl::lut16(v.x, v.y, v.z, v.w, e);
            float4* L = reinterpret_cast<float4*>(lut + (tid + k * kThreads) * kJs);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) L[i] = make_float4(e[4 * i], e[4 * i + 1], e[4 * i + 2], e[4 * i + 3]);
        }
    }
};

// -------------------------------------------------------------------------------------------------
// z worker for the in-kernel Mode 4 op: one side CTA = one unit (row m, ranks [rb*Rc, (rb+1)*Rc)).
// z[m][r] = S[r] * sum_k sign(V[r,k]) x[m][k] over the full K with a nibble LUT over x row m (in smem),
// each complete value published as one 8-byte (value, epoch) store (no atomics, no fence).
// All 256 threads call it. Rc in {8, 16, 32}; K/4 nibbles split over Q = 256/Rc threads per rank.
// -------------------------------------------------------------------------------------------------
template <int R>
CUTLASS_DEVICE void
bsvd_zworker(__nv_bfloat16 const* x, uint8_t const* v, __nv_bfloat16 const* s, int K, int Rc, int rb,
             float2* zt, int epoch, char* smem, int tid, int64_t* st, int copies, int copy_stride, int mode,
             int32_t const* ucol = nullptr, int n_cols = 0, uint32_t* pcorr = nullptr, int prow = 0, uint32_t* zt4 = nullptr) {
    // zt4 != nullptr: publish z as 4-byte values with an 8-bit tag ((epoch % 255) + 1) in the low mantissa bits
    const uint32_t ztag8 = static_cast<uint32_t>(epoch % 255) + 1u;
    using F = NibbleLut<1>;
    constexpr int kThreads = 256;
    const int r0 = rb * Rc;
    const int KW = K / 32;                      // V words per rank row
    const int lane = tid & 31;
    if (mode == 3 && Rc <= 8) {
        // ---- direct from bf16 (no conversion pass): WPR = 8 / Rc warps per rank, NS = 32 WPR slices; slice s owns
        // K nibbles c = s + NS j (interleaved: consecutive lanes read consecutive 8-byte x chunks, conflict-free).
        const int WPR = 8 / Rc;
        const int NS = 32 * WPR;
        const int NJ = (K / 4) / NS;            // nibbles per lane (5..40 at K = 5120)
        uint32_t* vrows = reinterpret_cast<uint32_t*>(smem);                                   // [Rc][KW]
        __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem + Rc * KW * 4);            // [K]
        float* red = reinterpret_cast<float*>(smem + Rc * KW * 4 + K * 2);                    // [8]
        {
            uint8_t const* vsrc = v + static_cast<size_t>(r0) * (K / 8);
            for (int c = tid; c < Rc * K / 128; c += kThreads) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(vrows + 4 * c)), "l"(vsrc + 16 * c) : "memory");
            }
            for (int c = tid; c < K / 8; c += kThreads) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(xs + 8 * c)), "l"(x + 8 * c) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        const int warp = tid >> 5;
        const int rr = warp / WPR;
        const int sl = (warp - rr * WPR) * 32 + lane;
        const float sv = (tid < Rc) ? bsvd_cl::ldg_bf16_early(s + r0 + tid) : 0.f;
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        if (st != nullptr && tid == 0) st[14] = bsvd_cl::gtimer();
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        {
            uint32_t const* vr = vrows + rr * KW;
            uint2 const* x2 = reinterpret_cast<uint2 const*>(xs);
            #pragma unroll 5
            for (int j = 0; j < NJ; ++j) {
                const int c = sl + NS * j;
                const uint2 w = x2[c];
                const uint32_t nb = ~(vr[c >> 3] >> ((c & 7) * 4));
                // element e: bf16 -> fp32 bits, sign bit xor'ed with the complemented V bit
                a0 += __uint_as_float((w.x << 16) ^ ((nb << 31) & 0x80000000u));
                a1 += __uint_as_float((w.x & 0xffff0000u) ^ ((nb << 30) & 0x80000000u));
                a2 += __uint_as_float((w.y << 16) ^ ((nb << 29) & 0x80000000u));
                a3 += __uint_as_float((w.y & 0xffff0000u) ^ ((nb << 28) & 0x80000000u));
            }
        }
        float a = (a0 + a1) + (a2 + a3);
        if (st != nullptr && tid == 0) st[9] = bsvd_cl::gtimer() + int64_t(__float_as_uint(a) & 0u);
        CUTLASS_PRAGMA_UNROLL
        for (int off = 16; off >= 1; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0) red[warp] = a;
        __syncthreads();
        if (st != nullptr && tid == 0) st[10] = bsvd_cl::gtimer();
        if (tid < Rc) {
            float z0 = 0.f, z1 = 0.f;
            CUTLASS_PRAGMA_UNROLL
            for (int w = 0; w < 8; w += 2) {
                if (w < WPR) z0 += red[tid * WPR + w];
                if (w + 1 < WPR) z1 += red[tid * WPR + w + 1];
            }
            if (zt4 != nullptr) {
                const uint32_t enc = (__float_as_uint((z0 + z1) * sv) & 0xFFFFFF00u) | ztag8;
                for (int c = 0; c < copies; ++c) __stcg(zt4 + static_cast<size_t>(c) * copy_stride + r0 + tid, enc);
            } else {
                const float2 val = make_float2((z0 + z1) * sv, __int_as_float(epoch));
                for (int c = 0; c < copies; ++c) __stcg(zt + static_cast<size_t>(c) * copy_stride + r0 + tid, val);
            }
        }
        if (st != nullptr && tid == 0) st[2] = bsvd_cl::gtimer();
        return;
    }
    if (mode == 1) {
        // ---- direct: WPR = 8 / Rc warps per rank; lane slice sl = (warp % WPR) * 32 + lane covers E = K / (32 WPR)
        // elements (E % 4 == 0), no LUT. x converted once to fp32 in smem, one segment per slice (segment stride an
        // odd number of float4: bank spread for LDS.128). Warp sums -> smem -> WPR-way add -> S -> publish.
        const int WPR = Rc <= 8 ? 8 / Rc : 1;    // warps per rank
        const int RPW = Rc <= 8 ? 1 : Rc / 8;    // ranks per warp (Rc > 8: each warp loops over RPW ranks)
        const int E = K / (32 * WPR);
        const int E4 = E / 4;                   // nibbles (float4) per slice
        const int ES4 = E4 | 1;                 // padded segment stride in float4
        const int NS = 32 * WPR;                // slices
        uint32_t* vrows = reinterpret_cast<uint32_t*>(smem);                                   // [Rc][KW]
        __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem + Rc * KW * 4);            // [K]
        float4* xf4 = reinterpret_cast<float4*>(smem + Rc * KW * 4 + K * 2);                  // [NS][ES4]
        float* red = reinterpret_cast<float*>(smem + Rc * KW * 4 + K * 2 + NS * ES4 * 16);   // [max(8, Rc)] warp / rank sums
        {
            uint8_t const* vsrc = v + static_cast<size_t>(r0) * (K / 8);
            for (int c = tid; c < Rc * K / 128; c += kThreads) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(vrows + 4 * c)), "l"(vsrc + 16 * c) : "memory");
            }
            for (int c = tid; c < K / 8; c += kThreads) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(xs + 8 * c)), "l"(x + 8 * c) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        const int warp = tid >> 5;
        const int rr0 = (warp / WPR) * RPW;     // first rank of this warp
        const int sl = (warp - (warp / WPR) * WPR) * 32 + lane;
        const float sv = (tid < Rc) ? bsvd_cl::ldg_bf16_early(s + r0 + tid) : 0.f;
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        if (st != nullptr && tid == 0) st[14] = bsvd_cl::gtimer();
        // conversion: all 256 threads, thread t = (slice t / P, part t % P), P = 256 / NS parts of E4 / P nibbles each
        {
            const int P = kThreads / NS;
            const int npp = E4 / P;
            const int slc = tid / P, part = tid - (tid / P) * P;
            uint2 const* src = reinterpret_cast<uint2 const*>(xs) + slc * E4 + part * npp;
            float4* dst = xf4 + slc * ES4 + part * npp;
            #pragma unroll 5
            for (int i = 0; i < npp; ++i) {
                const uint2 w4 = src[i];
                dst[i] = make_float4(bsvd_cl::bf_lo(w4.x), bsvd_cl::bf_hi(w4.x), bsvd_cl::bf_lo(w4.y), bsvd_cl::bf_hi(w4.y));
            }
        }
        __syncthreads();
        if (st != nullptr && tid == 0) st[1] = bsvd_cl::gtimer();
        #pragma unroll 1
        for (int rk = 0; rk < RPW; ++rk) {
        const int rr = rr0 + rk;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        {
            uint32_t const* vr = vrows + rr * KW;
            float4 const* xq = xf4 + sl * ES4;
            const int n0 = sl * E4;
            // x * (+1 if the V bit is set else -1) as an xor of the complemented bit into the sign bit:
            // shift + LOP3 + FADD per element, no predicates (predicate write-to-use latency is ~20 cycles)
            auto sgn = [](float xv, uint32_t t) -> float {       // t: complemented bit at position 31
                return __int_as_float(__float_as_int(xv) ^ static_cast<int>(t & 0x80000000u));
            };
            if ((E4 & 7) == 0) {
                // word-aligned slice (WPR = 1): one V word per 8 nibbles, constant shifts
                uint32_t const* vw = vr + (n0 >> 3);
                #pragma unroll 1
                for (int wi = 0; wi < E4 / 8; ++wi) {
                    const uint32_t nw = ~vw[wi];
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 8; ++i) {
                        const float4 xv = xq[wi * 8 + i];
                        a0 += sgn(xv.x, nw << (31 - 4 * i));
                        a1 += sgn(xv.y, nw << (30 - 4 * i));
                        a2 += sgn(xv.z, nw << (29 - 4 * i));
                        a3 += sgn(xv.w, nw << (28 - 4 * i));
                    }
                }
            } else {
                #pragma unroll 5
                for (int i = 0; i < E4; ++i) {
                    const int n = n0 + i;
                    const uint32_t nb = ~(vr[n >> 3] >> ((n & 7) * 4));
                    const float4 xv = xq[i];
                    a0 += sgn(xv.x, nb << 31);
                    a1 += sgn(xv.y, nb << 30);
                    a2 += sgn(xv.z, nb << 29);
                    a3 += sgn(xv.w, nb << 28);
                }
            }
        }
        float a = (a0 + a1) + (a2 + a3);
        if (st != nullptr && tid == 0 && rk == 0) st[9] = bsvd_cl::gtimer() + int64_t(__float_as_uint(a) & 0u);
        CUTLASS_PRAGMA_UNROLL
        for (int off = 16; off >= 1; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0) red[RPW > 1 ? rr : warp] = a;   // RPW > 1: one warp per rank group (WPR = 1), index by rank
        }
        __syncthreads();
        if (st != nullptr && tid == 0) st[10] = bsvd_cl::gtimer();
        if (tid < Rc) {
            float z0 = 0.f, z1 = 0.f;
            if (RPW > 1) {
                z0 = red[tid];
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < 8; w += 2) {
                    if (w < WPR) z0 += red[tid * WPR + w];
                    if (w + 1 < WPR) z1 += red[tid * WPR + w + 1];
                }
            }
            if (zt4 != nullptr) {
                const uint32_t enc = (__float_as_uint((z0 + z1) * sv) & 0xFFFFFF00u) | ztag8;
                for (int c = 0; c < copies; ++c) __stcg(zt4 + static_cast<size_t>(c) * copy_stride + r0 + tid, enc);
            } else {
                const float2 val = make_float2((z0 + z1) * sv, __int_as_float(epoch));
                for (int c = 0; c < copies; ++c) __stcg(zt + static_cast<size_t>(c) * copy_stride + r0 + tid, val);
            }
        }
        if (st != nullptr && tid == 0) st[2] = bsvd_cl::gtimer();
        return;
    }
    // ---- LUT: thread (q, rr) with rr fastest (lanes = ranks), full-K nibble LUT over x in smem. The thread's
    // V words (Nq/8 <= 20, contiguous in its rank row, 16-byte aligned) go straight to registers with 16-byte loads.
    constexpr int kMaxWords = 20;
    const int Q = kThreads / Rc;
    const int q = tid / Rc, rr = tid - (tid / Rc) * Rc;
    const int Nw = K / 4;
    const int Nq = Nw / Q;
    const int nwords = Nq / 8;
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);                                // [K]
    float* lut = reinterpret_cast<float*>(smem + K * 2);
    uint32_t vw[kMaxWords];
    const bool exact = (nwords == kMaxWords) && (Nw == 5 * kThreads);       // K = 5120, Rc = 32: no guards anywhere
    {
        uint4 const* vsrc = reinterpret_cast<uint4 const*>(v + static_cast<size_t>(r0 + rr) * (K / 8) + (q * Nq) / 2);
        if (exact) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kMaxWords / 4; ++i) {
                const uint4 w4 = bsvd_cl::ldg_v4_early(vsrc + i);
                vw[4 * i] = w4.x; vw[4 * i + 1] = w4.y; vw[4 * i + 2] = w4.z; vw[4 * i + 3] = w4.w;
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kMaxWords / 4; ++i) {
                const uint4 w4 = (4 * i < nwords) ? bsvd_cl::ldg_v4_early(vsrc + i) : make_uint4(0u, 0u, 0u, 0u);
                vw[4 * i] = w4.x; vw[4 * i + 1] = w4.y; vw[4 * i + 2] = w4.z; vw[4 * i + 3] = w4.w;
            }
        }
        for (int c = tid; c < K / 8; c += kThreads) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(xs + 8 * c)), "l"(x + 8 * c) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    // side-computed correction (pcorr): this unit's U word (32 ranks = word rb) of columns tid + 256 i
    constexpr int kMaxCols = 20;
    uint32_t ucw[kMaxCols];
    if (pcorr != nullptr) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kMaxCols; ++i) {
            const int n = tid + kThreads * i;
            ucw[i] = (n < n_cols) ? bsvd_cl::ldg_u32_early(ucol + static_cast<size_t>(n) * (R / 32) + rb) : 0u;
        }
    }
    const float sv = (tid < Rc) ? bsvd_cl::ldg_bf16_early(s + r0 + tid) : 0.f;
    auto xval4 = [&](int, int j) -> float4 {
        const uint2 wx = *reinterpret_cast<uint2 const*>(xs + 4 * j);
        return make_float4(bsvd_cl::bf_lo(wx.x), bsvd_cl::bf_hi(wx.x), bsvd_cl::bf_lo(wx.y), bsvd_cl::bf_hi(wx.y));
    };
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (st != nullptr && tid == 0) st[14] = bsvd_cl::gtimer();
    if (exact) F::template build_lut_exact<5>(lut, xval4, tid);
    else F::build_lut(lut, Nw, 1, xval4, tid);
    if (st != nullptr && tid == 0) st[8] = bsvd_cl::gtimer();
    __syncthreads();
    if (st != nullptr && tid == 0) st[1] = bsvd_cl::gtimer();
    float a;
    {
        // unguarded nibble lookups (as Fetch::lookup1), words from registers, fully unrolled
        const uint32_t lut_b = bsvd_cl::smem_u32(lut) + static_cast<uint32_t>(q * Nq * F::kJs * 4);
        constexpr int RB = F::kJs * 4;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int wi = 0; wi < kMaxWords; ++wi) {
            if (exact || wi < nwords) {
                const uint32_t w = vw[wi];
                const uint32_t we = w & 0x0F0F0F0Fu, wo = (w >> 4) & 0x0F0F0F0Fu;
                const uint32_t b = lut_b + static_cast<uint32_t>(wi * 8 * RB);
                const float t0 = bsvd_cl::lds_f32<0 * RB>(b + (__byte_perm(we, 0u, 0x4440u) << 2));
                const float t1 = bsvd_cl::lds_f32<1 * RB>(b + (__byte_perm(wo, 0u, 0x4440u) << 2));
                const float t2 = bsvd_cl::lds_f32<2 * RB>(b + (__byte_perm(we, 0u, 0x4441u) << 2));
                const float t3 = bsvd_cl::lds_f32<3 * RB>(b + (__byte_perm(wo, 0u, 0x4441u) << 2));
                const float t4 = bsvd_cl::lds_f32<4 * RB>(b + (__byte_perm(we, 0u, 0x4442u) << 2));
                const float t5 = bsvd_cl::lds_f32<5 * RB>(b + (__byte_perm(wo, 0u, 0x4442u) << 2));
                const float t6 = bsvd_cl::lds_f32<6 * RB>(b + (__byte_perm(we, 0u, 0x4443u) << 2));
                const float t7 = bsvd_cl::lds_f32<7 * RB>(b + (__byte_perm(wo, 0u, 0x4443u) << 2));
                a0 += t0 + t1; a1 += t2 + t3; a2 += t4 + t5; a3 += t6 + t7;
            }
        }
        a = (a0 + a1) + (a2 + a3);
    }
    if (st != nullptr && tid == 0) st[9] = bsvd_cl::gtimer() + int64_t(__float_as_uint(a) & 0u);
    __syncthreads();                                  // LUT reads done: reuse for the reduction
    float* red = lut;
    if (Rc < 32) {
        for (int off = 16; off >= Rc; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
    }
    if (lane < (Rc < 32 ? Rc : 32)) red[(tid >> 5) * 32 + lane] = a;
    __syncthreads();
    if (tid < Rc) {
        float s0 = 0.f, s1 = 0.f;
        if (Rc < 32) {
            CUTLASS_PRAGMA_UNROLL
            for (int ww = 0; ww < 8; ww += 2) { s0 += red[ww * 32 + tid]; s1 += red[(ww + 1) * 32 + tid]; }
        } else {
            const int wpr = Rc / 32;
            for (int qq = 0; qq < Q; qq += 2) {
                s0 += red[(qq * wpr + (tid >> 5)) * 32 + (tid & 31)];
                s1 += red[((qq + 1) * wpr + (tid >> 5)) * 32 + (tid & 31)];
            }
        }
        if (st != nullptr && tid == 0) st[10] = bsvd_cl::gtimer() + int64_t(__float_as_uint(s0 + s1) & 0u);
        const float zval = (s0 + s1) * sv;
        if (pcorr == nullptr && zt4 != nullptr) {
            const uint32_t enc = (__float_as_uint(zval) & 0xFFFFFF00u) | ztag8;
            for (int c = 0; c < copies; ++c) __stcg(zt4 + static_cast<size_t>(c) * copy_stride + r0 + tid, enc);
        } else if (pcorr == nullptr) {
            const float2 val = make_float2(zval, __int_as_float(epoch));
            for (int c = 0; c < copies; ++c) __stcg(zt + static_cast<size_t>(c) * copy_stride + r0 + tid, val);
        } else {
            red[512 + tid] = zval;                                   // z of this unit's 32 ranks -> smem
        }
    }
    if (pcorr != nullptr) {
        // partial correction of row prow over this unit's ranks for every column: 16-entry LUT per rank nibble
        // (8 nibbles), one lookup per nibble; published as 4-byte values with an 8-bit tag in the low mantissa
        // bits, layout [tile][rank block][row 16][64 columns]
        __syncthreads();
        float* zl = red + 576;                                        // [8][16]
        if (tid < 8) {
            float e[16];
            bsvd_cl::lut16(red[512 + 4 * tid], red[512 + 4 * tid + 1], red[512 + 4 * tid + 2], red[512 + 4 * tid + 3], e);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) reinterpret_cast<float4*>(zl + 16 * tid)[i] = make_float4(e[4 * i], e[4 * i + 1], e[4 * i + 2], e[4 * i + 3]);
        }
        __syncthreads();
        if (st != nullptr && tid == 0) st[10] = bsvd_cl::gtimer();
        const uint32_t tag = static_cast<uint32_t>(epoch % 255) + 1u;
        const int nblk = R / 32;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kMaxCols; ++i) {
            const int n = tid + kThreads * i;
            if (n < n_cols) {
                const uint32_t u = ucw[i];
                float c0 = 0.f, c1 = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 8; j += 2) {
                    c0 += zl[16 * j + ((u >> (4 * j)) & 15u)];
                    c1 += zl[16 * (j + 1) + ((u >> (4 * j + 4)) & 15u)];
                }
                const uint32_t enc = (__float_as_uint(c0 + c1) & 0xFFFFFF00u) | tag;
                const int t = n >> 6, c = n & 63;
                __stcg(pcorr + ((static_cast<size_t>(t) * nblk + rb) * 16 + prow) * 64 + c, enc);
            }
        }
    }
    if (st != nullptr && tid == 0) st[2] = bsvd_cl::gtimer();
}
