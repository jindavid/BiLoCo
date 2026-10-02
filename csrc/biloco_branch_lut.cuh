// BiLoCo binary branch for M > 1 with nibble lookup tables, run by persistent worker CTAs (256 threads).
// Included inside namespace biloco (biloco_gemm.cu), after biloco_side.cuh (uses its bsvd_cl helpers).
//
//   z[m][r]    = S[r] * sum_k sign(V[r][k]) x[m][k]          (phase Z)
//   corr[m][n] = sum_r z[m][r] sign(U[r][n])                 (phase C, after every z item is done)
//
// Rows are processed in m-tiles of 16. Both phases use a sign-symmetric 8-entry nibble LUT whose entry
// (nibble j, pattern p7) holds 16 row values (float4 x 4, 20-float stride: the 8 patterns of a nibble sit in
// 8 distinct bank groups): for a 4-bit sign pattern p (bit set = +1), sum_i s_i a_i = +L[p & 7] if bit 3
// is set, else -L[~p & 7]. One lookup = 4 LDS.128 + 16 FFMA for 16 rows x 4 ranks (or 4 k).
//
// Phase Z item = (m-tile, 32-rank block, K group): thread = (rank rr = lane, nibble slice q = warp); the
// x LUT is built per 64-nibble sub-chunk (40 KB); partial sums reduced over the 8 warps, scaled by S and
// atomically added into z[parity] (zeroed by the previous launch). One z-done counter (monotonic).
// Phase C item = (m-tile, 64-column GEMM tile): z LUT per m-tile (R/4 x 160 floats, kept across items of
// the same m-tile), thread = (column c, rank quarter h), quarters combined in smem, corr written fp32 as
// [tile][Mp][64]; one counter per tile (monotonic, +1 per m-tile).
namespace bsvd_br {

constexpr int kT = 256;
constexpr int kPS = 20;                 // floats per pattern (16 rows + 4 pad)
constexpr int kJS = 8 * kPS + 4;        // floats per nibble (+4: odd number of 16-byte chunks, so consecutive
                                        // LUT rows start in different bank groups; the build stores spread)
constexpr int kSub = 64;                // nibbles per x-LUT sub-chunk

struct Args {
    __nv_bfloat16 const* x;             // [M][K]
    uint8_t const* v;                   // [R][K/8]
    __nv_bfloat16 const* s;             // [R]
    int32_t const* ucol;                // [N][R/32]
    float* z;                           // [2][Mp][R] (parity; zero-initialised once by the host)
    float* corr;                        // [N/64][Mp][64]
    int* cnt;                           // [0] z items done; [1 + t] corr m-tiles done for tile t (all monotonic)
    int M, Mp, K, N, R, epoch, kg;
};

CUTLASS_DEVICE int ld_acquire(int const* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// 8 pattern values x 4 rows for one nibble: a[b][i] = value i of row b; e[p7][b]
CUTLASS_DEVICE void lut8_rows4(float const (&a)[4][4], float (&e)[8][4]) {
    CUTLASS_PRAGMA_UNROLL
    for (int b = 0; b < 4; ++b) {
        const float base = a[b][3] - a[b][0] - a[b][1] - a[b][2];
        const float d0 = 2.f * a[b][0], d1 = 2.f * a[b][1], d2 = 2.f * a[b][2];
        e[0][b] = base;         e[1][b] = base + d0;
        e[2][b] = base + d1;    e[3][b] = e[1][b] + d1;
        e[4][b] = base + d2;    e[5][b] = e[1][b] + d2;
        e[6][b] = e[2][b] + d2; e[7][b] = e[3][b] + d2;
    }
}

// acc[16] += sign-symmetric LUT lookup of nibble pattern pn at LUT row `row` (16 rows)
CUTLASS_DEVICE void lut_acc16(float const* lut, int row, uint32_t pn, float (&acc)[16]) {
    const uint32_t t = (pn >> 3) - 1u;                       // 0 if bit 3 set, else all ones
    const uint32_t p7 = (pn ^ t) & 7u;
    const float sg = __int_as_float(0x3f800000 ^ (t & 0x80000000u));
    float4 const* L = reinterpret_cast<float4 const*>(lut + row * kJS + static_cast<int>(p7) * kPS);
    CUTLASS_PRAGMA_UNROLL
    for (int qq = 0; qq < 4; ++qq) {
        const float4 l = L[qq];
        acc[4 * qq + 0] = fmaf(sg, l.x, acc[4 * qq + 0]);
        acc[4 * qq + 1] = fmaf(sg, l.y, acc[4 * qq + 1]);
        acc[4 * qq + 2] = fmaf(sg, l.z, acc[4 * qq + 2]);
        acc[4 * qq + 3] = fmaf(sg, l.w, acc[4 * qq + 3]);
    }
}

// Phase Z stages a whole item's operands in smem with one cp.async wave: x [16 rows][4 * nib_item (+8)] bf16
// and V [32 ranks][nib_item / 2 (+16)] bytes; nib_item <= kMaxNibItem (host: kg >= 1280 / kMaxNibItem).
constexpr int kMaxNibItem = 640;
constexpr int kXStride = 4 * kMaxNibItem + 8;          // bf16 per staged x row (+16 bytes: rows in different banks)
constexpr int kVStride = kMaxNibItem / 2 + 16;         // bytes per staged V row (16-byte aligned rows)
constexpr int kZSmem = 16 * kXStride * 2 + 32 * kVStride + kSub * kJS * 4 + 8 * 16 * 32 * 4;

// smem needed by one worker CTA (bytes)
CUTLASS_HOST_DEVICE constexpr int smem_bytes(int R) {
    return kZSmem > ((R / 4) * kJS * 4 + 4 * 16 * 64 * 4) ? kZSmem : ((R / 4) * kJS * 4 + 4 * 16 * 64 * 4);
}

// Run both phases as worker `w` of `nw` workers. All workers must be co-resident (phase C waits for
// every z item). st: optional globaltimer stamps [0] start, [1] z done (own), [2] z complete seen, [3] end.
CUTLASS_DEVICE void run(Args const& a, char* smem, int w, int nw, int64_t* st) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31, warp = tid >> 5;
    const int par = a.epoch & 1;
    const int n_mt = a.Mp / 16;
    const int n_rb = a.R / 32;
    const int nib_total = a.K / 4;
    const int nib_item = nib_total / a.kg;                  // multiple of kSub, <= kMaxNibItem (host-checked)
    const int n_zitems = n_mt * n_rb * a.kg;
    float* zcur = a.z + static_cast<size_t>(par) * a.Mp * a.R;
    float* zoth = a.z + static_cast<size_t>(par ^ 1) * a.Mp * a.R;
    if (st != nullptr && tid == 0) st[0] = bsvd_cl::gtimer();
    // zero the other parity for the next launch (not read in this launch)
    for (int i = w * kT + tid; i < a.Mp * a.R; i += nw * kT) zoth[i] = 0.f;

    // ---------------- phase Z
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);                               // [16][kXStride]
    uint8_t* vs = reinterpret_cast<uint8_t*>(smem + 16 * kXStride * 2);                         // [32][kVStride]
    float* lut = reinterpret_cast<float*>(smem + 16 * kXStride * 2 + 32 * kVStride);            // [kSub][kJS]
    float* red = lut + kSub * kJS;                                                              // [8][16][32]
    const int jb = tid >> 2, mq = tid & 3;                  // build task: nibble jb of the sub-chunk, rows 4 mq .. +3
    for (int it = w; it < n_zitems; it += nw) {
        const int g = it % a.kg, rb = (it / a.kg) % n_rb, mt = it / (a.kg * n_rb);
        const int m0 = mt * 16, r0 = rb * 32, nib0 = g * nib_item;
        const int mrows = (a.M - m0) < 16 ? (a.M - m0) : 16;
        // stage x rows m0 .. m0 + mrows - 1 (k in [4 nib0, 4 (nib0 + nib_item))) and 32 V rows, one wave
        __syncthreads();                                   // previous item done with xs / vs / red
        {
            const int xch = nib_item / 2;                  // 16-byte chunks per x row
            for (int c = tid; c < mrows * xch; c += kT) {
                const int m = c / xch, cc = c - m * xch;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(xs + m * kXStride + 8 * cc)),
                             "l"(a.x + static_cast<size_t>(m0 + m) * a.K + 4 * nib0 + 8 * cc) : "memory");
            }
            const int vch = nib_item / 32;                 // 16-byte chunks per V row
            for (int c = tid; c < 32 * vch; c += kT) {
                const int r = c / vch, cc = c - r * vch;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(vs + r * kVStride + 16 * cc)),
                             "l"(a.v + static_cast<size_t>(r0 + r) * (a.K / 8) + nib0 / 2 + 16 * cc) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_all;" ::: "memory");
        }
        __syncthreads();
        float acc[16];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i) acc[i] = 0.f;
        #pragma unroll 1
        for (int sc = 0; sc < nib_item / kSub; ++sc) {
            float xa[4][4];
            CUTLASS_PRAGMA_UNROLL
            for (int b = 0; b < 4; ++b) {
                const int m = 4 * mq + b;
                if (m < mrows) {
                    const uint2 q2 = *reinterpret_cast<uint2 const*>(xs + m * kXStride + 4 * (sc * kSub + jb));
                    xa[b][0] = bsvd_cl::bf_lo(q2.x); xa[b][1] = bsvd_cl::bf_hi(q2.x);
                    xa[b][2] = bsvd_cl::bf_lo(q2.y); xa[b][3] = bsvd_cl::bf_hi(q2.y);
                } else {
                    xa[b][0] = xa[b][1] = xa[b][2] = xa[b][3] = 0.f;
                }
            }
            float e[8][4];
            lut8_rows4(xa, e);
            const uint32_t vw = *reinterpret_cast<uint32_t const*>(vs + lane * kVStride + (sc * kSub + warp * 8) / 2);
            if (sc > 0) __syncthreads();                   // previous sub-chunk's lookups done
            CUTLASS_PRAGMA_UNROLL
            for (int p7 = 0; p7 < 8; ++p7) {
                *reinterpret_cast<float4*>(lut + jb * kJS + p7 * kPS + 4 * mq) = make_float4(e[p7][0], e[p7][1], e[p7][2], e[p7][3]);
            }
            __syncthreads();
            CUTLASS_PRAGMA_UNROLL
            for (int jj = 0; jj < 8; ++jj) lut_acc16(lut, warp * 8 + jj, (vw >> (4 * jj)) & 15u, acc);
        }
        // reduce over the 8 warps (nibble slices)
        __syncthreads();
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < 16; ++row) red[(warp * 16 + row) * 32 + lane] = acc[row];
        __syncthreads();
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < 2; ++k) {
            const int o = tid + k * kT, row = o >> 5, rr = o & 31;
            float sum = 0.f;
            CUTLASS_PRAGMA_UNROLL
            for (int q = 0; q < 8; ++q) sum += red[(q * 16 + row) * 32 + rr];
            if (row < mrows) {
                atomicAdd(zcur + static_cast<size_t>(m0 + row) * a.R + r0 + rr, sum * __bfloat162float(a.s[r0 + rr]));
            }
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicAdd(a.cnt, 1);
    }
    if (st != nullptr && tid == 0) st[1] = bsvd_cl::gtimer();

    // ---------------- phase C: worker w owns m-tile w % n_mt (z LUT built once) and tiles w / n_mt + k * wpm
    const int n_tiles = a.N / 64;
    const int wpm = nw / n_mt;
    const bool c_active = w < wpm * n_mt;
    const int mt = w % n_mt, m0 = mt * 16;
    const int c = tid & 63, h = tid >> 6;
    const int wq = a.R / 128;                                                 // U words per rank quarter
    constexpr int kMaxWq = 4;                                                 // R <= 512
    uint32_t uw[kMaxWq];
    auto load_u = [&](int t) {
        const int n = t * 64 + c;
        CUTLASS_PRAGMA_UNROLL
        for (int wi = 0; wi < kMaxWq; ++wi) {
            uw[wi] = (wi < wq) ? static_cast<uint32_t>(__ldg(a.ucol + static_cast<size_t>(n) * (a.R / 32) + h * wq + wi)) : 0u;
        }
    };
    int t = w / n_mt;
    if (!c_active || t >= n_tiles) {                                          // no correction item: done
        if (st != nullptr && tid == 0) { st[2] = bsvd_cl::gtimer(); st[3] = st[2]; }
        return;
    }
    load_u(t);                                                                // overlaps the wait for z

    // ---------------- wait for every z item of this launch
    if (tid == 0) {
        const int target = a.epoch * n_zitems;
        while (ld_acquire(a.cnt) < target) __nanosleep(64);
    }
    __syncthreads();
    if (st != nullptr && tid == 0) st[2] = bsvd_cl::gtimer();

    float* zlut = reinterpret_cast<float*>(smem);                             // [R/4][kJS]
    float* cred = reinterpret_cast<float*>(smem + (a.R / 4) * kJS * 4);       // [4][16][64]
    for (int task = tid; task < a.R; task += kT) {
        const int j = task >> 2, q4 = task & 3;
        float za[4][4];
        CUTLASS_PRAGMA_UNROLL
        for (int b = 0; b < 4; ++b) {
            const float4 z4 = __ldcg(reinterpret_cast<float4 const*>(zcur + static_cast<size_t>(m0 + 4 * q4 + b) * a.R + 4 * j));
            za[b][0] = z4.x; za[b][1] = z4.y; za[b][2] = z4.z; za[b][3] = z4.w;
        }
        float e[8][4];
        lut8_rows4(za, e);
        CUTLASS_PRAGMA_UNROLL
        for (int p7 = 0; p7 < 8; ++p7) {
            *reinterpret_cast<float4*>(zlut + j * kJS + p7 * kPS + 4 * q4) = make_float4(e[p7][0], e[p7][1], e[p7][2], e[p7][3]);
        }
    }
    __syncthreads();
    for (; t < n_tiles; t += wpm) {
        float acc[16];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i) acc[i] = 0.f;
        uint32_t uc[kMaxWq];
        CUTLASS_PRAGMA_UNROLL
        for (int wi = 0; wi < kMaxWq; ++wi) uc[wi] = uw[wi];
        if (t + wpm < n_tiles) load_u(t + wpm);                               // next item's U words in flight
        CUTLASS_PRAGMA_UNROLL
        for (int wi = 0; wi < kMaxWq; ++wi) {
            if (wi < wq) {
                const int wd = h * wq + wi;
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < 8; ++jj) lut_acc16(zlut, wd * 8 + jj, (uc[wi] >> (4 * jj)) & 15u, acc);
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < 16; ++row) cred[(h * 16 + row) * 64 + c] = acc[row];
        __syncthreads();
        float* dst = a.corr + (static_cast<size_t>(t) * a.Mp + m0) * 64;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < 4; ++k) {
            const int o = tid + k * kT, row = o >> 6, cc = o & 63;
            const float v = (cred[(0 * 16 + row) * 64 + cc] + cred[(1 * 16 + row) * 64 + cc])
                          + (cred[(2 * 16 + row) * 64 + cc] + cred[(3 * 16 + row) * 64 + cc]);
            __stcg(dst + row * 64 + cc, v);
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicAdd(a.cnt + 1 + t, 1);
    }
    if (st != nullptr && tid == 0) st[3] = bsvd_cl::gtimer();
}

// ------------------------------------------------------------------------------------------------------------
// Mode 4 side z worker for small M (2 <= M <= kRows, kRows in {4, 8}): one side CTA = one rank block
// [rb*RB, rb*RB + RB) for ALL rows. x rows and the block's V rows are staged once (cp.async); K is walked in
// sub-chunks of KS nibbles with a kRows-row sign-symmetric LUT (pattern stride kPSr floats, nibble stride kJSr:
// odd numbers of 16-byte chunks spread the build stores and the lookups over the bank groups). Thread =
// (rank rr = tid % RB, slice q = tid / RB); slice q owns nibbles q + Q i of each sub-chunk. Sums reduced over
// the slices (shuffles, then smem across warps), scaled by S, published as (value, epoch) float2 into every
// z copy: zt[c * copy_stride + m * R + r].
template <int kRows>
struct RowsCfg {
    static constexpr int kQ4 = kRows / 4;
    static constexpr int kPSr = kRows == 4 ? 4 : 12;
    static constexpr int kJSr = 8 * kPSr + 4;
    static constexpr int KS = kRows == 4 ? 640 : 128;   // nibbles per sub-chunk (4 rows: 2 sub-chunks at K = 5120)
};

template <int kRows>
CUTLASS_HOST_DEVICE constexpr int zrows_smem(int K, int RB) {
    return kRows * K * 2 + RB * (K / 8) + RowsCfg<kRows>::KS * RowsCfg<kRows>::kJSr * 4 + 8 * kRows * 8 * 4 + 64;
}

template <int kRows>
CUTLASS_DEVICE void zworker_rows(__nv_bfloat16 const* x, uint8_t const* v, __nv_bfloat16 const* s, int M, int K, int R,
                                 int RB, int rb, float2* zt, int epoch, char* smem, int tid, int copies, int copy_stride,
                                 int64_t* st = nullptr) {
    using C = RowsCfg<kRows>;
    constexpr int KS = C::KS;
    const int r0 = rb * RB;
    const int Q = kT / RB;
    const int rr = tid % RB, q = tid / RB;
    const int lane = tid & 31, warp = tid >> 5;
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);                           // [kRows][K]
    uint8_t* vs = reinterpret_cast<uint8_t*>(smem + kRows * K * 2);                         // [RB][K/8]
    float* lut = reinterpret_cast<float*>(smem + kRows * K * 2 + RB * (K / 8));              // [KS][kJSr]
    float* red = lut + KS * C::kJSr;                                                         // [8 warps][kRows][8]
    {
        for (int c = tid; c < M * (K / 8); c += kT) {
            const int m = c / (K / 8), cc = c - m * (K / 8);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(xs + m * K + 8 * cc)),
                         "l"(x + static_cast<size_t>(m) * K + 8 * cc) : "memory");
        }
        uint8_t const* vsrc = v + static_cast<size_t>(r0) * (K / 8);
        for (int c = tid; c < RB * (K / 128); c += kT) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(vs + 16 * c)), "l"(vsrc + 16 * c) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    // S of this thread's output rank (outputs: tid < M * RB, rank tid % RB), issued before the wait
    const float sv = (tid < M * RB) ? bsvd_cl::ldg_bf16_early(s + r0 + tid % RB) : 0.f;
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (st != nullptr && tid == 0) st[14] = bsvd_cl::gtimer();
    float acc[kRows];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kRows; ++i) acc[i] = 0.f;
    const int jpt = KS / Q;                                  // nibbles per thread per sub-chunk (1, 2 or 4)
    #pragma unroll 1
    for (int sc = 0; sc < (K / 4) / KS; ++sc) {
        // build: task = (nibble j, row quad q4), j fastest
        for (int t = tid; t < KS * C::kQ4; t += kT) {
            const int j = t % KS, q4 = t / KS;
            float xa[4][4];
            CUTLASS_PRAGMA_UNROLL
            for (int b = 0; b < 4; ++b) {
                const int m = 4 * q4 + b;
                if (m < M) {
                    const uint2 q2 = *reinterpret_cast<uint2 const*>(xs + m * K + 4 * (sc * KS + j));
                    xa[b][0] = bsvd_cl::bf_lo(q2.x); xa[b][1] = bsvd_cl::bf_hi(q2.x);
                    xa[b][2] = bsvd_cl::bf_lo(q2.y); xa[b][3] = bsvd_cl::bf_hi(q2.y);
                } else {
                    xa[b][0] = xa[b][1] = xa[b][2] = xa[b][3] = 0.f;
                }
            }
            float e[8][4];
            lut8_rows4(xa, e);
            CUTLASS_PRAGMA_UNROLL
            for (int p7 = 0; p7 < 8; ++p7) {
                *reinterpret_cast<float4*>(lut + j * C::kJSr + p7 * C::kPSr + 4 * q4) = make_float4(e[p7][0], e[p7][1], e[p7][2], e[p7][3]);
            }
        }
        __syncthreads();
        // lookups in batches of 5 (jpt = 5, 10 or 20): offsets / signs, then the loads, then the FMAs
        #pragma unroll 1
        for (int i0 = 0; i0 < jpt; i0 += 5) {
            int off[5];
            float sg[5];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < 5; ++k) {
                const int j = q + Q * (i0 + k);
                const int ng = sc * KS + j;
                const uint32_t pn = (static_cast<uint32_t>(vs[rr * (K / 8) + (ng >> 1)]) >> ((ng & 1) * 4)) & 15u;
                const uint32_t tt = (pn >> 3) - 1u;
                off[k] = j * C::kJSr + static_cast<int>((pn ^ tt) & 7u) * C::kPSr;
                sg[k] = __int_as_float(0x3f800000 ^ (tt & 0x80000000u));
            }
            float4 l[5][C::kQ4];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < 5; ++k) {
                CUTLASS_PRAGMA_UNROLL
                for (int q4 = 0; q4 < C::kQ4; ++q4) l[k][q4] = reinterpret_cast<float4 const*>(lut + off[k])[q4];
            }
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < 5; ++k) {
                CUTLASS_PRAGMA_UNROLL
                for (int q4 = 0; q4 < C::kQ4; ++q4) {
                    acc[4 * q4 + 0] = fmaf(sg[k], l[k][q4].x, acc[4 * q4 + 0]);
                    acc[4 * q4 + 1] = fmaf(sg[k], l[k][q4].y, acc[4 * q4 + 1]);
                    acc[4 * q4 + 2] = fmaf(sg[k], l[k][q4].z, acc[4 * q4 + 2]);
                    acc[4 * q4 + 3] = fmaf(sg[k], l[k][q4].w, acc[4 * q4 + 3]);
                }
            }
        }
        __syncthreads();
    }
    if (st != nullptr && tid == 0) st[9] = bsvd_cl::gtimer();
    // reduce over the slices: lanes with the same rank inside a warp, then the 8 warps
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kRows; ++i) {
        for (int off = 16; off >= RB; off >>= 1) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], off);
    }
    if (lane < RB) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kRows; ++i) red[(warp * kRows + i) * 8 + lane] = acc[i];
    }
    __syncthreads();
    if (tid < M * RB) {
        const int m = tid / RB, r = tid - (tid / RB) * RB;
        float z0 = 0.f, z1 = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < 8; w += 2) { z0 += red[(w * kRows + m) * 8 + r]; z1 += red[((w + 1) * kRows + m) * 8 + r]; }
        const float2 val = make_float2((z0 + z1) * sv, __int_as_float(epoch));
        for (int c = 0; c < copies; ++c) __stcg(zt + static_cast<size_t>(c) * copy_stride + static_cast<size_t>(m) * R + r0 + r, val);
    }
    if (st != nullptr && tid == 0) st[2] = bsvd_cl::gtimer();
}

// Direct z for small M (no LUT): one side CTA = ranks [rb*Rc, (rb+1)*Rc) x all M rows (M <= kR; rows M..kR-1 zero).
// As the M = 1 bf16-direct worker: WPR = 8 / Rc warps per rank, slice sl = (warp % WPR) * 32 + lane owns the K
// nibbles c = sl + 32 WPR j; the complemented V nibble gives the 4 sign masks once, shared by the kR rows.
CUTLASS_HOST_DEVICE constexpr int zdirect_smem(int K, int Rc, int kR) {
    return Rc * (K / 8) + kR * K * 2 + 8 * kR * 4;
}

template <int kR>
CUTLASS_DEVICE void zworker_direct_rows(__nv_bfloat16 const* x, uint8_t const* v, __nv_bfloat16 const* s, int M, int K, int R,
                                        int Rc, int rb, float2* zt, int epoch, char* smem, int tid, int copies, int copy_stride,
                                        int64_t* st = nullptr) {
    const int r0 = rb * Rc;
    const int KW = K / 32;
    const int lane = tid & 31, warp = tid >> 5;
    const int WPR = 8 / Rc, NS = 32 * WPR, NJ = (K / 4) / NS;
    uint32_t* vrows = reinterpret_cast<uint32_t*>(smem);                                   // [Rc][KW]
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem + Rc * KW * 4);            // [kR][K]
    float* red = reinterpret_cast<float*>(smem + Rc * KW * 4 + kR * K * 2);               // [8 warps][kR]
    {
        uint8_t const* vsrc = v + static_cast<size_t>(r0) * (K / 8);
        for (int c = tid; c < Rc * K / 128; c += kT) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(vrows + 4 * c)), "l"(vsrc + 16 * c) : "memory");
        }
        for (int c = tid; c < M * (K / 8); c += kT) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(xs + 8 * c)), "l"(x + 8 * c) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        for (int c = M * (K / 8) + tid; c < kR * (K / 8); c += kT) {
            *reinterpret_cast<uint4*>(xs + 8 * c) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    const int rr = warp / WPR;
    const int sl = (warp - rr * WPR) * 32 + lane;
    const float sv = (tid < M * Rc) ? bsvd_cl::ldg_bf16_early(s + r0 + tid % Rc) : 0.f;
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (st != nullptr && tid == 0) st[14] = bsvd_cl::gtimer();
    float a[kR][4];
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kR; ++m) a[m][0] = a[m][1] = a[m][2] = a[m][3] = 0.f;
    {
        uint32_t const* vr = vrows + rr * KW;
        uint2 const* x2 = reinterpret_cast<uint2 const*>(xs);
        #pragma unroll 4
        for (int j = 0; j < NJ; ++j) {
            const int c = sl + NS * j;
            const uint32_t nb = ~(vr[c >> 3] >> ((c & 7) * 4));
            const uint32_t s0 = nb << 31, s1 = (nb << 30) & 0x80000000u, s2 = (nb << 29) & 0x80000000u, s3 = (nb << 28) & 0x80000000u;
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < kR; ++m) {
                const uint2 w = x2[m * (K / 4) + c];
                a[m][0] += __uint_as_float((w.x << 16) ^ s0);
                a[m][1] += __uint_as_float((w.x & 0xffff0000u) ^ s1);
                a[m][2] += __uint_as_float((w.y << 16) ^ s2);
                a[m][3] += __uint_as_float((w.y & 0xffff0000u) ^ s3);
            }
        }
    }
    if (st != nullptr && tid == 0) st[9] = bsvd_cl::gtimer() + int64_t(__float_as_uint(a[0][0]) & 0u);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kR; ++m) {
        float t = (a[m][0] + a[m][1]) + (a[m][2] + a[m][3]);
        CUTLASS_PRAGMA_UNROLL
        for (int off = 16; off >= 1; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0) red[warp * kR + m] = t;
    }
    __syncthreads();
    if (tid < M * Rc) {
        const int m = tid / Rc, r = tid - (tid / Rc) * Rc;
        float z = 0.f;
        for (int w = 0; w < WPR; ++w) z += red[(r * WPR + w) * kR + m];
        const float2 val = make_float2(z * sv, __int_as_float(epoch));
        for (int c = 0; c < copies; ++c) __stcg(zt + static_cast<size_t>(c) * copy_stride + static_cast<size_t>(m) * R + r0 + r, val);
    }
    if (st != nullptr && tid == 0) st[2] = bsvd_cl::gtimer();
}

// Rank-inner direct z for small M: one side CTA = kRc ranks x all M rows (M <= kR; rows M..kR-1 zero). Thread t owns
// the K nibbles c = t + 256 j: its x nibble (kR rows) is converted once and reused by the kRc ranks; the 4 signs of a
// (rank, nibble) come from one LDS.128 of a 16-entry +-1 table, then kR x 4 FFMA. Transposed smem reduction.
CUTLASS_HOST_DEVICE constexpr int zinner_smem(int K, int kRc, int kR) {
    return kRc * (K / 8) + kR * K * 2 + 16 * 16 + kRc * kR * 288 * 4;
}

template <int kR, int kRc, int kUnrollNF = 5>
CUTLASS_DEVICE void zworker_ranks_inner(__nv_bfloat16 const* x, uint8_t const* v, __nv_bfloat16 const* s, int M, int K, int R,
                                        int rb, float2* zt, int epoch, char* smem, int tid, int copies, int copy_stride,
                                        int64_t* st = nullptr, bool f2 = false) {
    constexpr int kNO = kRc * kR;                  // outputs per CTA
    constexpr int kTPO = (kT / kNO) < 32 ? (kT / kNO) : 32;   // threads (lanes of one warp) per output
    constexpr int kRS = kT + kTPO;                 // partials stride per output (conflict-free strided reads)
    static_assert((kTPO & (kTPO - 1)) == 0 && kNO * kTPO <= kT, "reduction shape");
    const int r0 = rb * kRc;
    const int KW = K / 32;
    uint32_t* vrows = reinterpret_cast<uint32_t*>(smem);                                   // [kRc][KW]
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem + kRc * KW * 4);           // [kR][K]
    float4* sgn = reinterpret_cast<float4*>(smem + kRc * KW * 4 + kR * K * 2);           // [16]
    float* red = reinterpret_cast<float*>(sgn + 16);                                      // [kNO][kRS]
    const int NJ = (K / 4) / kT;                   // nibble iterations (5 at K = 5120)
    {
        // one commit group (a 5-chunk pipeline with a barrier per chunk measured slower: data 0.4 us later)
        uint8_t const* vsrc = v + static_cast<size_t>(r0) * (K / 8);
        for (int c = tid; c < kRc * K / 128; c += kT) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(vrows + 4 * c)), "l"(vsrc + 16 * c) : "memory");
        }
        for (int c = tid; c < M * (K / 8); c += kT) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(bsvd_cl::smem_u32(xs + 8 * c)), "l"(x + 8 * c) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        for (int c = M * (K / 8) + tid; c < kR * (K / 8); c += kT) {
            *reinterpret_cast<uint4*>(xs + 8 * c) = make_uint4(0u, 0u, 0u, 0u);
        }
        if (tid < 16) {
            sgn[tid] = make_float4((tid & 1) ? 1.f : -1.f, (tid & 2) ? 1.f : -1.f, (tid & 4) ? 1.f : -1.f, (tid & 8) ? 1.f : -1.f);
        }
    }
    // S of the output this thread publishes (tid = o * kTPO, o = m * kRc + r), issued before the wait
    const int o = tid / kTPO, part = tid % kTPO;
    const int om = o / kRc, orr = o - (o / kRc) * kRc;
    // every lane of an output's group publishes some of the z copies (the xor-shuffle leaves the sum in all of them)
    const bool pub = o < kNO && om < M;
    const float sv = pub ? bsvd_cl::ldg_bf16_early(s + r0 + orr) : 0.f;
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (st != nullptr && tid == 0) st[14] = bsvd_cl::gtimer();
    float a[kRc][kR];
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kRc; ++r) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kR; ++m) a[r][m] = 0.f;
    }
    if constexpr (kR >= 4) {
        if (f2) {
            // packed fp32x2 FMA over row pairs (m, m + 1): half the FMA instructions
            auto pk2 = [](float lo, float hi) -> unsigned long long {
                return static_cast<unsigned long long>(__float_as_uint(lo)) | (static_cast<unsigned long long>(__float_as_uint(hi)) << 32);
            };
            unsigned long long c2[kRc][kR / 2];
            CUTLASS_PRAGMA_UNROLL
            for (int r = 0; r < kRc; ++r) {
                CUTLASS_PRAGMA_UNROLL
                for (int mp = 0; mp < kR / 2; ++mp) c2[r][mp] = 0ull;
            }
            const int sh = (tid & 7) * 4;
            uint2 const* x2 = reinterpret_cast<uint2 const*>(xs) + tid;
            uint32_t const* vw = vrows + (tid >> 3);
            #pragma unroll 1
            for (int j = 0; j < NJ; ++j) {
                unsigned long long xp[kR / 2][4];
                CUTLASS_PRAGMA_UNROLL
                for (int mp = 0; mp < kR / 2; ++mp) {
                    const uint2 wa = x2[(2 * mp) * (K / 4) + kT * j];
                    const uint2 wb = x2[(2 * mp + 1) * (K / 4) + kT * j];
                    xp[mp][0] = pk2(bsvd_cl::bf_lo(wa.x), bsvd_cl::bf_lo(wb.x));
                    xp[mp][1] = pk2(bsvd_cl::bf_hi(wa.x), bsvd_cl::bf_hi(wb.x));
                    xp[mp][2] = pk2(bsvd_cl::bf_lo(wa.y), bsvd_cl::bf_lo(wb.y));
                    xp[mp][3] = pk2(bsvd_cl::bf_hi(wa.y), bsvd_cl::bf_hi(wb.y));
                }
                CUTLASS_PRAGMA_UNROLL
                for (int r = 0; r < kRc; ++r) {
                    const uint32_t nib = (vw[r * KW + (kT / 8) * j] >> sh) & 15u;
                    const float4 g = sgn[nib];
                    const unsigned long long g0 = pk2(g.x, g.x), g1 = pk2(g.y, g.y), g2 = pk2(g.z, g.z), g3 = pk2(g.w, g.w);
                    CUTLASS_PRAGMA_UNROLL
                    for (int mp = 0; mp < kR / 2; ++mp) {
                        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(c2[r][mp]) : "l"(g0), "l"(xp[mp][0]));
                        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(c2[r][mp]) : "l"(g1), "l"(xp[mp][1]));
                        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(c2[r][mp]) : "l"(g2), "l"(xp[mp][2]));
                        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(c2[r][mp]) : "l"(g3), "l"(xp[mp][3]));
                    }
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int r = 0; r < kRc; ++r) {
                CUTLASS_PRAGMA_UNROLL
                for (int mp = 0; mp < kR / 2; ++mp) {
                    a[r][2 * mp] = __uint_as_float(static_cast<uint32_t>(c2[r][mp]));
                    a[r][2 * mp + 1] = __uint_as_float(static_cast<uint32_t>(c2[r][mp] >> 32));
                }
            }
        }
    }
    if (!(kR >= 4 && f2)) {
        const int sh = (tid & 7) * 4;
        uint2 const* x2 = reinterpret_cast<uint2 const*>(xs) + tid;
        uint32_t const* vw = vrows + (tid >> 3);
        // 8 rows: a rolled loop (the 5x unrolled body is instruction-fetch bound on side SMs behind busy L2 paths)
        constexpr int kUnroll = kR * kRc >= 64 ? 1 : kUnrollNF;
        #pragma unroll kUnroll
        for (int j = 0; j < NJ; ++j) {
            float xf[kR][4];
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < kR; ++m) {
                const uint2 w = x2[m * (K / 4) + kT * j];
                xf[m][0] = bsvd_cl::bf_lo(w.x); xf[m][1] = bsvd_cl::bf_hi(w.x);
                xf[m][2] = bsvd_cl::bf_lo(w.y); xf[m][3] = bsvd_cl::bf_hi(w.y);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int r = 0; r < kRc; ++r) {
                const uint32_t nib = (vw[r * KW + (kT / 8) * j] >> sh) & 15u;
                const float4 g = sgn[nib];
                CUTLASS_PRAGMA_UNROLL
                for (int m = 0; m < kR; ++m) {
                    a[r][m] = fmaf(g.x, xf[m][0], a[r][m]);
                    a[r][m] = fmaf(g.y, xf[m][1], a[r][m]);
                    a[r][m] = fmaf(g.z, xf[m][2], a[r][m]);
                    a[r][m] = fmaf(g.w, xf[m][3], a[r][m]);
                }
            }
        }
    }
    if (st != nullptr && tid == 0) st[9] = bsvd_cl::gtimer() + int64_t(__float_as_uint(a[0][0]) & 0u);
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kRc; ++r) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kR; ++m) red[(m * kRc + r) * kRS + tid] = a[r][m];
    }
    __syncthreads();
    if (st != nullptr && tid == 0) st[10] = bsvd_cl::gtimer();
    // output o = m * kRc + r summed by kTPO lanes of one warp, each over kT / kTPO strided partials
    if (o < kNO) {
        float z0 = 0.f, z1 = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kT / kTPO; i += 2) {
            z0 += red[o * kRS + part + kTPO * i];
            z1 += red[o * kRS + part + kTPO * (i + 1)];
        }
        float z = z0 + z1;
        if (st != nullptr && tid == 0) st[11] = bsvd_cl::gtimer() + int64_t(__float_as_uint(z) & 0u);
        CUTLASS_PRAGMA_UNROLL
        for (int off = kTPO / 2; off >= 1; off >>= 1) z += __shfl_xor_sync(0xffffffffu, z, off);
        if (st != nullptr && tid == 0) st[12] = bsvd_cl::gtimer() + int64_t(__float_as_uint(z) & 0u);
        if (pub) {
            const float2 val = make_float2(z * sv, __int_as_float(epoch));
            for (int c = part; c < copies; c += kTPO) __stcg(zt + static_cast<size_t>(c) * copy_stride + static_cast<size_t>(om) * R + r0 + orr, val);
        }
    }
    if (st != nullptr && tid == 0) st[2] = bsvd_cl::gtimer();
}

}  // namespace bsvd_br
