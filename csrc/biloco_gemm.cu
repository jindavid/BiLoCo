#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/functional.h"
#include "cutlass/arch/barrier.h"
#include <cstdlib>
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/packed_stride.hpp"

#include "helper.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "element_traits.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <typeinfo>
#include <c10/util/TypeCast.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace biloco
{
    using namespace cute;

    // BiLoCo epilogue visitor. The four epilogue store warps (128 threads) of the stock SM100
    // warp-specialized GEMM are idle during the mainloop; their begin() hook runs before the
    // accumulator wait, so all low-rank work there overlaps the tensor-core mainloop. The GEMM grid
    // has one CTA per 64-column output tile (N/64 <= SM count, all co-resident); the patched CUTLASS
    // kernel adds side CTAs on the remaining SMs that never touch the GEMM (binarysvd_side_hook.hpp).
    //   Mode 2: side CTAs compute z[m][i] = S_i * sum_k sign(V_ik) x[m][k] in units of side_rc ranks
    //           (V in a permuted nibble layout, K = 5120) into epoch-tagged scratch and bump an arrival
    //           counter (Flags[0]); tiles wait for it, then expand (M <= 32).
    //   Mode 4: side CTAs compute complete z values and publish (value, epoch) pairs; tiles poll the
    //           tags (M <= 16, any K).
    //   Mode 5: every CTA runs the nibble-LUT branch of biloco_branch_lut.cuh (z, then the correction) at
    //           kernel entry, then the tiles run the GEMM and add their slice (large M x R).
    //   Mode 6: Mode 2 with dense BF16 factors (the equal-rank control).
    // The expansion corr[m][n] = sum_i z[m][i] u[i][n] uses 4-bit LUTs per (row, rank nibble) in smem;
    // the result is added to the FP32 accumulators before the BF16 store.
#include "biloco_side.cuh"
#include "biloco_branch_lut.cuh"

    template <class ElementCompute, int RankStatic, int TileN, int MaxRows, int Mode>
    struct Sm90BinaryZUCwFetch {
        static constexpr int kRankWords = (RankStatic + 31) / 32;
        static constexpr int kNibbles = RankStatic / 4;
        static constexpr int kTileN = TileN;
        static constexpr int kMaxRows = MaxRows;
        static constexpr int kThreads = 128;
        static constexpr int kBarZ = 1;      // tile path: the epilogue warps' own named barrier (EpilogueBarrier)
        static constexpr int kBarW = 1;      // side CTAs use raw ids 12/13 (no other warps run CUTLASS code there)
#define BSVD_TILE_HELPERS 0   // helper warps: correct but slower (sched warp is held by the CLC throttle until ~3.5-4 us)
        // Mode 4 tile helper warps (sched + idle epilogue-load warps) take the last rank words of the single-pass
        // expansions (2-row LUT; 16-entry 4-row LUT at r >= 512); the epilogue warps' two rank halves take kHelpA words each
        static constexpr bool kHelp = (BSVD_TILE_HELPERS != 0) && Mode == 4 && (MaxRows == 2 || (MaxRows == 4 && RankStatic >= 512));
        static constexpr int kHelpA0 = (kRankWords + 2) / 3;
        static constexpr int kHelpA = kHelp ? (kHelpA0 + ((MaxRows == 4) ? (kHelpA0 % 2) : 0)) : kRankWords / 2;   // 4-row: even (swizzle)
        static constexpr int kHelpW = kRankWords - 2 * kHelpA;                     // helper words
        // lean tile code for the 2- and 8-row Mode 4 kernels: only the variants they run are compiled (the dead
        // alternatives sat in the fetched post-poll path of these instruction-fetch-bound tiles)
        static constexpr bool kLeanTile = (Mode == 4 && (MaxRows == 2 || MaxRows == 8));
        static constexpr bool kV2Only = (Mode == 4 && MaxRows == 2 && RankStatic >= 512);   // 2-row: v2 expansion at r >= 512
        static constexpr bool kV1Only = (Mode == 4 && MaxRows == 2 && RankStatic < 512);    // 2-row: v1 below
        static constexpr int kMaxChunk = 8;  // max ranks per tile (host-checked)
        static constexpr int kNb = 5;        // 16-byte x slots per thread per K pass (5120 K per pass)
        static constexpr int kKPass = kThreads * kNb * 8;   // 5120
        static constexpr int kRowBatch = MaxRows < 2 ? MaxRows : 2;   // rows per z unit (register budget: no spills)
        static constexpr int kRcMax = 32;                                // max ranks per side unit (v3)
        static constexpr int kSideLutBytes = 16 * 7 * 128 * 4;           // group 0 per-thread nibble LUTs [7 nibbles][16 patterns][128 threads] fp32
        static constexpr int kSideRedBytes = 2 * 4 * kRcMax * 4 + 64;    // [group][warp][rank] partials + [8 warps] row max
        CUTLASS_HOST_DEVICE static constexpr int side_group_bytes(int rc) { return kSideLutBytes + rc * 1024 + kSideRedBytes; }   // per side CTA (both groups on one unit)
        static constexpr int kStage = (Mode == 2 || Mode == 6) ? (MaxRows == 1 ? RankStatic : (RankStatic < 128 ? RankStatic : 128)) : 64;   // ranks staged per pass (Mode 2)
        static constexpr int kStagePad = kStage + 4;                                  // compact fp32 row stride (M > 1): rows of a warp land in different banks
        static constexpr int kDiagTab = 4;
#define BSVD_POLL_DEEP 0   // 4-deep rings (<= 4 float4 per thread) measured slower (more L2 poll traffic): off
        static constexpr int kPollDeep = BSVD_POLL_DEEP;                               // Mode 4 poll: 4-deep ring up to this many float4 per thread
        static constexpr int kLutRows = 1;
        static constexpr int kNibbles16 = 16;
        static constexpr bool kInKernelZ = (Mode == 2 || Mode == 6);
        static_assert(TileN == 64, "cw visitor assumes a 64-wide N tile");
        static_assert(RankStatic % 16 == 0, "rank must be a multiple of 16");

        struct SharedStorage {
            cute::array_aligned<ElementCompute, (Mode == 5 ? 1 : (kHelp ? 3 : 2)) * kMaxRows * kTileN> smem_part;    // [half][row][col]; half 0 = final (Mode 5: [row][col])
            cute::array_aligned<uint16_t, ((Mode == 2 || Mode == 4 || Mode == 5 || Mode == 6) ? 2 : kMaxRows * RankStatic)> smem_z16;   // z as fp16 [row][rank] (mma expansion; Mode 1)
            cute::array_aligned<ElementCompute, ((Mode == 2 || Mode == 6) ? kMaxRows * kStage * 2 : (MaxRows == 1 ? (RankStatic > 64 ? RankStatic : 64) : ((Mode == 4 && MaxRows == 4 && RankStatic >= 512) ? 4 : (Mode == 4 && MaxRows <= 8) ? MaxRows * RankStatic : ((Mode == 4 || Mode == 5) ? 4 : MaxRows * 64))))> smem_z32;  // Mode 2: tagged z pairs [rows][kStage] (TMA bulk copy target) / Mode 1: full row
            cute::array_aligned<ElementCompute, ((Mode == 2 || Mode == 5 || Mode == 6) ? 4 : (MaxRows == 1 ? RankStatic * 4 : ((Mode == 4 && MaxRows <= 8) ? (MaxRows == 4 ? (RankStatic >= 512 ? (RankStatic / 4) * 64 : (RankStatic / 4) * 36) : (MaxRows == 2 ? (RankStatic / 4) * 36 : 32 * 100)) : (Mode == 4 ? RankStatic / 4 * 8 * 20 : 4))))> smem_lut4; // Mode 4 M>1: z LUT [R/4 nibbles][8 patterns][20 (16 rows + pad)]
            cute::array_aligned<ElementCompute, kDiagTab> smem_lut;                   // diag scratch only
            cute::array_aligned<ElementCompute, 4 * kRowBatch * kMaxChunk> smem_red;  // [warp][row][rank] partials
            cute::array_aligned<uint32_t, ((Mode == 2 || Mode == 6) ? (MaxRows == 1 ? kMaxChunk * 128 * 2 : kRankWords * 64) : 2)> smem_vw;  // Mode 2: M=1 U nibble LUTs (2048 fp32) / M>1 U tile [words][64] (TMA); else unused
            cute::array_aligned<int32_t, 64> smem_diag;
        };

        struct Arguments {
            ElementCompute* z_ptr = nullptr;            // Mode 1: [m_rows][RankStatic] fp32. Mode 2: [m_rows][RankStatic] float2 (value, epoch tag)
            int32_t const* u_col32_ptr = nullptr;       // [n_cols][kRankWords]
            int rank = 0;
            int rank_words = 0;
            int n_cols = 0;
            int m_rows = 0;
            // Mode 2 only
            __nv_bfloat16 const* x_ptr = nullptr;       // [m_rows][k_cols]
            uint8_t const* v_ptr = nullptr;             // [RankStatic][k_cols/8]
            __nv_bfloat16 const* s_ptr = nullptr;       // [RankStatic]
            int32_t* flags_ptr = nullptr;               // [0]: arrival counter (release-add per tile; target = epoch * num_workers)
            int k_cols = 0;
            int num_workers = 0;                        // = number of N tiles (all co-resident)
            int epoch = 0;
            int num_side = 0;                           // side CTAs (extra grid layer); 0 = tiles compute z themselves
            int side_kind = 0;                          // 0: two 128-thread select/add units per side CTA; 1: one mma item per side CTA
            int n_units = 0;                            // arrival-counter increments per launch
            int side_ks = 1;                            // mma side path: K split factor (items x ks active side CTAs)
            int pdl_wait = 0;                           // Mode 1: wait on the primary grid (z kernel, PDL) before staging z
            uint8_t const* vperm_ptr = nullptr;         // V pre-permuted [R][2 words][128 threads]: word0 = nibbles 0..6 (bits 4j), word1 = nibbles 7..9 (bits 4(j-7)); nibble j of thread t = v bits 4*(128j+t)..+3
            int tile_dp2a = 0;                          // M > 1 tile expansion: 0 predicated fp32 adds, 1 int16 z x int8 u dot products (dp2a)
            int side_mode = 0;                          // side unit: 0 = LUT (6 nibbles) + FFMA (4 nibbles); 1 = FFMA only (no LUT build; used for small rc)
            int32_t const* u_rw_ptr = nullptr;          // U word-major [kRankWords][n_cols] (coalesced tile loads); null -> u_col32_ptr
            int side_rc = 8;                            // ranks per side unit (8 | 16 | 32)
            int side_groups = 2;                        // 128-thread groups per side CTA
            int poll_mode = 0;                          // (unused) tile poll variant
            int z_copies = 8;                           // side units publish z into z_copies replicas [copy][m_rows][R]; tile reads copy tile % z_copies
            int dbg_flags = 0;                          // bit 0: tiles skip the z wait + expansion (timing only); bit 1: side CTAs idle
            int m2_warm = 0;                            // Mode 2, M = 1: warm-up pass of the expansion before the poll
            int zw_rc = 8;                              // Mode 4: ranks per side z worker
            int poll_gap_ns = 250;                      // Mode 4: spacing of the pipelined z polls
            int tile_lut = 0;                           // Mode 4, M = 1: 1 = nibble-LUT expansion (16-entry, smem_lut4) instead of selects
            int tile_warm = 0;                          // Mode 4: run the expansion once on stale smem before polling (instruction-cache warm-up)
            int side_corr = 0;                          // Mode 4, M > 1: side units publish partial corrections; tiles only sum them
            int exp2 = 1;                               // Mode 4, M > 1: expansion variant 2 (thread = column x 8 rows, no combine)
            int z4b = 1;                                // Mode 4: z published as 4-byte values with an 8-bit tag in the low mantissa bits
            int exp2r = 1;                              // Mode 4, M <= 2 in the 4-row kernel: 2-row 16-entry LUT expansion
            int side_rot = 0;                           // Mode 4 zw_mode 6: rank block = (side_idx + side_rot) % units (diagnostic)
            int side_f2 = 0;                            // Mode 4 zw_mode 6, M > 2: packed fp32x2 FMA side loop
            int exp4r16 = 1;                            // Mode 4, 3-4 rows in the 4-row kernel: 16-entry 4-row LUT expansion
            int poll_delay_ns = 0;                      // Mode 4: delay before the tiles start polling z (after begin())
            int poll_anchor_ns = 0;                     // Mode 4: ring slot k is not issued before tile CTA entry + anchor + k * anchor_gap
            int poll_anchor_gap_ns = 150;
            int helpers = 0;                            // Mode 4 (kHelp kernels): tile helper warps active (must be 1 there)
            int warm_cheap = 0;                         // Mode 4, 2-row kernel: cheap warm pass (instruction fetch only) before the poll
            int exp2v2 = 0;                             // Mode 4, 2-row kernel: v2 LUT expansion (packed offsets, shuffle half-combine; needs warm_cheap)
            int side_unroll = 5;                        // Mode 4 rank-inner side worker, M <= 2: nibble-loop unroll (1, 2 or 5)
            int zw_mode = 0;                            // Mode 4 side z worker: 0 = x-nibble LUT (units of Rc ranks, lanes = ranks), 1 = direct sign-add (warp = rank)
            static constexpr bool kBinarySvdSide = true;
            CUTLASS_DEVICE static void side_cta(Arguments const& p, char* smem, int side_idx);
            // Mode 5 (M > 1, one launch): every CTA (tiles first, then side CTAs) runs the LUT branch
            // (bsvd_branch_lut.cuh) at kernel entry; tiles then run the GEMM and add their correction slice.
            bsvd_br::Args br{};
            int br_tiles = 0;
            int br_tiles_work = 1;                      // Mode 5: tiles also run branch items at entry (workers numbered sides first)
            // which helper expansion a tile runs: 0 none, 1 2-row LUT, 2 16-entry 4-row LUT (epilogue and helpers agree)
            CUTLASS_DEVICE static int help_path(Arguments const& p) {
                if constexpr (!kHelp) return 0;
                if (!p.helpers) return 0;
                if (MaxRows == 2 || (p.m_rows <= 2 && p.exp2r)) return 1;
                return p.exp4r16 ? 2 : 0;
            }
            // tile helper warps (hw 0: sched warp, 1: epilogue-load warp): columns lane + 32 hw, rank words
            // [2 kHelpA, kRankWords); named barrier 12 = LUT ready (128 epilogue arrive, 64 helpers sync),
            // 13 = helper partials in smem_part rows [2 MaxRows, 3 MaxRows) (64 arrive, 128 sync)
            CUTLASS_DEVICE static void tile_helper(Arguments const& p, void* fs, int tile_n, int hw) {
                if constexpr (kHelp) {
                    const int path = help_path(p);
                    if (path == 0) return;
                    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(fs);
                    float const* lut = ss.smem_lut4.data();
                    float* part = ss.smem_part.data();
                    const int lane = static_cast<int>(threadIdx.x & 31);
                    const int c = lane + 32 * hw;
                    const int n = tile_n * kTileN + c;
                    const bool okn = n < p.n_cols;
                    if (p.diag_ptr != nullptr && lane == 0) p.diag_ptr[tile_n * 8 + 5 + hw] = globaltimer();
                    uint32_t uw[kHelpW > 0 ? kHelpW : 1];
                    CUTLASS_PRAGMA_UNROLL
                    for (int w = 0; w < kHelpW; ++w) uw[w] = okn ? bsvd_cl::ldg_u32_early(p.u_col32_ptr + static_cast<size_t>(n) * kRankWords + 2 * kHelpA + w) : 0u;
                    asm volatile("bar.sync 12, 192;" ::: "memory");
                    if (path == 1) {
                        constexpr int kJS = 36;
                        float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
                        CUTLASS_PRAGMA_UNROLL
                        for (int w = 0; w < kHelpW; ++w) {
                            uint32_t u = uw[w];
                            asm volatile("" : "+r"(u));
                            float2 l[8];
                            CUTLASS_PRAGMA_UNROLL
                            for (int q = 0; q < 8; ++q) {
                                const int j = 8 * (2 * kHelpA + w) + q;
                                l[q] = *reinterpret_cast<float2 const*>(lut + j * kJS + 2 * static_cast<int>((u >> (4 * q)) & 15u));
                            }
                            CUTLASS_PRAGMA_UNROLL
                            for (int q = 0; q < 8; q += 2) { a0 += l[q].x; a1 += l[q].y; b0 += l[q + 1].x; b1 += l[q + 1].y; }
                        }
                        part[(2 * kMaxRows) * kTileN + c] = a0 + b0;
                        part[(2 * kMaxRows + 1) * kTileN + c] = a1 + b1;
                    } else {
                        float4 const* lut4 = reinterpret_cast<float4 const*>(lut);
                        float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = make_float4(0.f, 0.f, 0.f, 0.f);
                        CUTLASS_PRAGMA_UNROLL
                        for (int w = 0; w < kHelpW; ++w) {
                            uint32_t u = uw[w];
                            asm volatile("" : "+r"(u));
                            float4 l[8];
                            CUTLASS_PRAGMA_UNROLL
                            for (int q = 0; q < 8; ++q) {
                                const int j = 8 * (2 * kHelpA + w) + q;
                                l[q] = lut4[j * 16 + static_cast<int>(((u >> (4 * q)) & 15u) ^ static_cast<uint32_t>(j & 15))];
                            }
                            CUTLASS_PRAGMA_UNROLL
                            for (int q = 0; q < 8; q += 2) {
                                a.x += l[q].x; a.y += l[q].y; a.z += l[q].z; a.w += l[q].w;
                                b.x += l[q + 1].x; b.y += l[q + 1].y; b.z += l[q + 1].z; b.w += l[q + 1].w;
                            }
                        }
                        if constexpr (MaxRows >= 4) {
                            part[(2 * kMaxRows) * kTileN + c] = a.x + b.x;
                            part[(2 * kMaxRows + 1) * kTileN + c] = a.y + b.y;
                            part[(2 * kMaxRows + 2) * kTileN + c] = a.z + b.z;
                            part[(2 * kMaxRows + 3) * kTileN + c] = a.w + b.w;
                        }
                    }
                    asm volatile("bar.arrive 13, 192;" ::: "memory");
                }
            }
            // tile CTA entry (fork hook, before any pipeline init): Mode 4 stamps the entry time into smem_vw[0..1]
            // (unused in Mode 4) for the poll anchor
            CUTLASS_DEVICE static void tile_entry(Arguments const& p, void* fs, int tile_idx) {
                if constexpr (Mode == 4) {
                    if (threadIdx.x == 0) {   // unconditional: no params load at kernel entry (a cold constant-cache miss delays every warp)
                        int64_t t;
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                        uint32_t* vw = reinterpret_cast<SharedStorage*>(fs)->smem_vw.data();
                        vw[0] = static_cast<uint32_t>(t);
                        vw[1] = static_cast<uint32_t>(static_cast<uint64_t>(t) >> 32);
                    }
                }
                (void)tile_idx;
            }
            CUTLASS_DEVICE static void tile_pre(Arguments const& p, char* smem, int tile_idx, int n_tiles) {
                if constexpr (Mode == 5) {
                    if (p.br_tiles_work) {
                        bsvd_br::run(p.br, smem, p.num_side + tile_idx, p.num_side + n_tiles, nullptr);
                        __syncthreads();
                    }
                }
            }
            // Mode 0 diagnostics
            int spin_iters = 0;                         // dependent FMAs per thread in begin()
            int chase_iters = 0;                        // dependent global loads per thread in begin()
            int32_t const* chase_ptr = nullptr;
            int64_t* diag_ptr = nullptr;                // [cta][4] globaltimer stamps: ctor, begin, first visit, end
        };

        using Params = Arguments;

        template <class ProblemShape>
        static constexpr Params
        to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
            return args;
        }

        template <class ProblemShape>
        static bool
        can_implement(ProblemShape const&, Arguments const& args) {
            bool ok = args.u_col32_ptr != nullptr &&
                      args.rank == RankStatic && args.rank_words == kRankWords &&
                      args.n_cols > 0 && args.m_rows > 0 && args.m_rows <= kMaxRows;
            if constexpr (Mode == 2 || Mode == 4 || Mode == 6) {
                ok = ok && args.z_ptr != nullptr;
            }
            if constexpr (kInKernelZ) {
                ok = ok && args.x_ptr != nullptr && args.v_ptr != nullptr &&
                     args.s_ptr != nullptr && args.flags_ptr != nullptr &&
                     args.k_cols > 0 && (args.k_cols % 8) == 0 && args.num_workers > 0;
            }
            return ok;
        }

        template <class ProblemShape>
        static size_t
        get_workspace_size(ProblemShape const&, Arguments const&) {
            return 0;
        }

        template <class ProblemShape>
        static cutlass::Status
        initialize_workspace(ProblemShape const&, Arguments const&, void*,
                             cudaStream_t, cutlass::CudaHostAdapter* = nullptr) {
            return cutlass::Status::kSuccess;
        }

        CUTLASS_DEVICE static int64_t globaltimer() {
            int64_t t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            return t;
        }

        CUTLASS_HOST_DEVICE
        Sm90BinaryZUCwFetch() { }

        CUTLASS_HOST_DEVICE
        Sm90BinaryZUCwFetch(Params const& params, SharedStorage const& shared_storage)
            : params_ptr(&params),
              smem_part(const_cast<ElementCompute*>(shared_storage.smem_part.data())),
              smem_z16(const_cast<uint16_t*>(shared_storage.smem_z16.data())),
              smem_z32((Mode == 4 && MaxRows == 4 && RankStatic >= 512) ? const_cast<ElementCompute*>(shared_storage.smem_lut4.data()) + 12 * RankStatic
                                                  : const_cast<ElementCompute*>(shared_storage.smem_z32.data())),   // Mode 4, 4 rows: z [4][R] in the LUT tail
              smem_lut4(const_cast<ElementCompute*>(shared_storage.smem_lut4.data())),
              smem_lut(const_cast<ElementCompute*>(shared_storage.smem_lut.data())),
              smem_red(const_cast<ElementCompute*>(shared_storage.smem_red.data())),
              smem_vw(const_cast<uint32_t*>(shared_storage.smem_vw.data())) {
#if defined(__CUDA_ARCH__)
            if constexpr (Mode == 2) {
                if (params.diag_ptr != nullptr && threadIdx.x == 128) {
                    const int64_t t = globaltimer();
                    atomicMin(reinterpret_cast<unsigned long long*>(params.diag_ptr + 4092), static_cast<unsigned long long>(t));
                    atomicMax(reinterpret_cast<unsigned long long*>(params.diag_ptr + 4093), static_cast<unsigned long long>(t));
                    params.diag_ptr[4089] = int64_t(gridDim.x) | (int64_t(gridDim.y) << 16) | (int64_t(gridDim.z) << 32);
                    atomicMax(reinterpret_cast<unsigned long long*>(params.diag_ptr + 4086), static_cast<unsigned long long>(blockIdx.y));
                    const_cast<SharedStorage&>(shared_storage).smem_diag[0] = static_cast<int32_t>(t & 0x7fffffff);
                }
            }
#endif
        }

        Params const* params_ptr;
        ElementCompute* smem_part = nullptr;
        uint16_t* smem_z16 = nullptr;
        ElementCompute* smem_z32 = nullptr;
        ElementCompute* smem_lut4 = nullptr;
        ElementCompute* smem_lut = nullptr;
        ElementCompute* smem_red = nullptr;
        uint32_t* smem_vw = nullptr;

        CUTLASS_DEVICE bool
        is_producer_load_needed() const {
            return false;
        }

        CUTLASS_DEVICE bool
        is_C_load_needed() const {
            return false;
        }

        template <class... Args>
        CUTLASS_DEVICE auto
        get_producer_load_callbacks(
            cutlass::epilogue::fusion::ProducerLoadArgs<Args...> const&) {
            return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
        }

        CUTLASS_DEVICE static void bar_sync(int id) {
            asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(kThreads) : "memory");
        }

        template <class CTensor, class ProblemShapeMNL>
        struct ConsumerStoreCallbacks
            : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
            CUTLASS_DEVICE
            ConsumerStoreCallbacks(CTensor&& coords,
                                   ProblemShapeMNL problem_shape_mnl,
                                   Params const* params_ptr,
                                   ElementCompute* smem_part,
                                   uint16_t* smem_z16,
                                   ElementCompute* smem_z32,
                                   ElementCompute* smem_lut4,
                                   ElementCompute* smem_lut,
                                   ElementCompute* smem_red,
                                   uint32_t* smem_vw,
                                   int tile_m_start,
                                   int tile_n_start,
                                   int tile_n_idx,
                                   int thread_idx)
                : coords(cute::forward<CTensor>(coords)),
                  problem_shape_mnl(problem_shape_mnl),
                  params_ptr(params_ptr),
                  smem_part(smem_part),
                  smem_z16(smem_z16),
                  smem_z32(smem_z32),
                  smem_lut4(smem_lut4),
                  smem_lut(smem_lut),
                  smem_red(smem_red),
                  smem_vw(smem_vw),
                  tile_m_start(tile_m_start),
                  tile_n_start(tile_n_start),
                  tile_n_idx(tile_n_idx),
                  thread_idx(thread_idx) {
                rows_valid = params_ptr->m_rows - tile_m_start;
                rows_valid = rows_valid > kMaxRows ? kMaxRows : rows_valid;
                if constexpr (Mode == 4) diag4 = params_ptr->diag_ptr;   // cached: no params loads after the mainloop
            }

            CTensor coords;
            int64_t* diag4 = nullptr;
            ProblemShapeMNL problem_shape_mnl;
            Params const* params_ptr;
            ElementCompute* smem_part;
            uint16_t* smem_z16;
            ElementCompute* smem_z32;
            ElementCompute* smem_lut4;
            ElementCompute* smem_lut;
            ElementCompute* smem_red;
            uint32_t* smem_vw;
            int tile_m_start;
            int tile_n_start;
            int tile_n_idx;
            int thread_idx;
            int rows_valid;
            bool stamped_visit = false;
            // Mode 4, 2-row kernel: cheap warm pass (pass 0) runs the expansion code with the data-dependent smem address
            // parts masked to 0 (am_ = 0) and no smem writes (wr_ = false): the post-poll code is fetched into the
            // instruction cache before z lands (tile SMs fetch cold code behind the weight stream)
            int am_ = -1;
            bool wr_ = true;
            int my_row = -1;          // this thread's local row within the tile (fixed for the T2R layout)
            int sub_n0 = 0;           // column of fragment 0 of the current subtile (relative to tile)
            bool my_row_live = false;

            CUTLASS_DEVICE bool
            begin_sync_needed() const {
                return true;
            }

            // Diagnostic stamp (Mode 0/2): slot j of this tile, relative to the CTA ctor.
            CUTLASS_DEVICE void stamp(int j) {
#define BSVD_STAMPS 0   // diagnostic stamps: off in the timing build (params loads + branches in the hot paths)
                if constexpr (Mode == 4 && BSVD_STAMPS == 0) return;
                if constexpr (Mode == 4) {
                    if (j >= 5) return;   // Mode 4: begin() stamps only (no params loads after the mainloop)
                }
                if constexpr (Mode == 2 || Mode == 4) {
                    if (params_ptr->diag_ptr != nullptr && thread_idx == 0) {
                        params_ptr->diag_ptr[tile_n_idx * 8 + j] = globaltimer();
                        if constexpr (Mode == 4) {
                            if (j == 0) {
                                uint32_t smid;
                                asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
                                params_ptr->diag_ptr[tile_n_idx * 8 + 7] = smid;
                            }
                        }
                    }
                }
            }

            CUTLASS_DEVICE static float4 ld_cg_volatile(float4 const* p) {
                float4 v;
                asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
                return v;
            }
            CUTLASS_DEVICE static void mma_bf16_16816(float (&d)[4], uint32_t const (&a)[4], uint32_t const (&b)[2]) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
            CUTLASS_DEVICE static void mma_f16_16816(float (&d)[4], uint32_t const (&a)[4], uint32_t const (&b)[2]) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
            // two packed +-1 values from two sign bits (bit set = +1): bf16 (+1 = 0x3F80) or f16 (+1 = 0x3C00)
            CUTLASS_DEVICE static uint32_t sign_pair_bf16(uint32_t bit_lo, uint32_t bit_hi) {
                return 0x3F803F80u | ((bit_lo ^ 1u) << 15) | ((bit_hi ^ 1u) << 31);
            }
            CUTLASS_DEVICE static uint32_t sign_pair_f16(uint32_t bit_lo, uint32_t bit_hi) {
                return 0x3C003C00u | ((bit_lo ^ 1u) << 15) | ((bit_hi ^ 1u) << 31);
            }

            // One z work unit: (row group of kRowBatch rows) x (rank chunk of <= 8 ranks) computed by
            // 128 threads (tid), reduced across its 4 warps through `red`, scaled by S, stored to
            // global z (fp32), then one release-add on the arrival counter. Used by tiles
            // (num_side == 0) and by side CTAs (extra grid layer).
            CUTLASS_DEVICE static void
            z_unit(Params const& P, int rows, int worker, int num_workers, int tid, float* red, uint32_t* vw_smem, int bar_id, int64_t* st = nullptr) {
                const int n_rg = (rows + kRowBatch - 1) / kRowBatch;
                const int T_g = num_workers / n_rg;                 // tiles per row group (host: >= 1)
                const int rg = worker / T_g;
                const int ci = worker - rg * T_g;
                const int chunk = (RankStatic + T_g - 1) / T_g;
                const int ib = ci * chunk;
                const int ie = (ib + chunk) < RankStatic ? (ib + chunk) : RankStatic;
                const int m0 = rg * kRowBatch;
                for (int c0 = ib; c0 < ie; c0 += kMaxChunk) {
                const int i0 = c0;
                const int i1 = (c0 + kMaxChunk) < ie ? (c0 + kMaxChunk) : ie;
                const int nch = (rg < n_rg && i1 > i0) ? (i1 - i0) : 0;
                const int mrows = (rg < n_rg) ? ((rows - m0) < kRowBatch ? (rows - m0) : kRowBatch) : 0;
                (void)mrows;
                const int warp = tid >> 5;
                const int lane = tid & 31;
                const int kb = P.k_cols >> 3;
                const int K = P.k_cols;
                const int vtotal = RankStatic * kb;
                constexpr int kRed = kRowBatch * kMaxChunk;
                // ---- issue every global load up front (S, V words, x rows), consume afterwards
                uint16_t s_raw[kMaxChunk];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kMaxChunk; ++i) {
                    s_raw[i] = (i < nch) ? __ldg(reinterpret_cast<uint16_t const*>(P.s_ptr) + i0 + i) : uint16_t(0);
                }
                float sv[kMaxChunk];
                if (nch > 0 && mrows > 0) {
                    float acc[kRowBatch][kMaxChunk];
                    CUTLASS_PRAGMA_UNROLL
                    for (int mm = 0; mm < kRowBatch; ++mm) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kMaxChunk; ++i) acc[mm][i] = 0.f;
                    }
                    for (int k0 = 0; k0 < K; k0 += kKPass) {
                        const int kbase = k0 + tid * (kNb * 8);
                        const int boff = kbase >> 3;
                        // V words for this thread's 40 k of every rank in the group -> smem (per unit: [kMaxChunk][128][2])
                        uint32_t* vsm = vw_smem;
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kMaxChunk; ++i) {
                            uint32_t w0 = 0u, w1 = 0u, sh = 0u;
                            if (i < nch && boff < kb) {
                                const int off = (i0 + i) * kb + boff;
                                const int a = off & ~3;
                                sh = uint32_t(off & 3) * 8u;
                                w0 = __ldg(reinterpret_cast<uint32_t const*>(P.v_ptr + a));
                                if (a + 8 <= vtotal) {
                                    w1 = __ldg(reinterpret_cast<uint32_t const*>(P.v_ptr + a + 4));
                                } else {
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int b = 0; b < 4; ++b) {
                                        if (a + 4 + b < vtotal) w1 |= uint32_t(__ldg(P.v_ptr + a + 4 + b)) << (8 * b);
                                    }
                                }
                            }
                            const uint64_t v64 = ((uint64_t(w1) << 32) | uint64_t(w0)) >> sh;   // 40 valid bits
                            vsm[(i * kThreads + tid) * 2] = uint32_t(v64);
                            vsm[(i * kThreads + tid) * 2 + 1] = uint32_t(v64 >> 32);
                        }
                        uint4 xr[kRowBatch][kNb];
                        CUTLASS_PRAGMA_UNROLL
                        for (int mm = 0; mm < kRowBatch; ++mm) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int j = 0; j < kNb; ++j) {
                                const int k = kbase + 8 * j;
                                xr[mm][j] = (mm < mrows && k < K)
                                    ? __ldg(reinterpret_cast<uint4 const*>(P.x_ptr + static_cast<size_t>(m0 + mm) * K + k))
                                    : make_uint4(0u, 0u, 0u, 0u);
                            }
                        }
                        if (st != nullptr && tid == 0 && c0 == ib) {
                            if ((vsm[tid * 2] ^ xr[0][0].x ^ uint32_t(s_raw[0])) == 0x7fc00001u) st[7] = -1;
                            st[1] = globaltimer();
                        }
                        float xf[kRowBatch][kNb * 8];
                        CUTLASS_PRAGMA_UNROLL
                        for (int mm = 0; mm < kRowBatch; ++mm) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int j = 0; j < kNb; ++j) {
                                const uint32_t xw[4] = {xr[mm][j].x, xr[mm][j].y, xr[mm][j].z, xr[mm][j].w};
                                CUTLASS_PRAGMA_UNROLL
                                for (int q = 0; q < 4; ++q) {
                                    xf[mm][8 * j + 2 * q] = __uint_as_float(xw[q] << 16);
                                    xf[mm][8 * j + 2 * q + 1] = __uint_as_float(xw[q] & 0xffff0000u);
                                }
                            }
                        }
                        __syncwarp();
                        // hot loop over ranks (small code): 40 select/add per row per rank
                        #pragma unroll 1
                        for (int i = 0; i < nch; ++i) {
                            const uint32_t vlo = vsm[(i * kThreads + tid) * 2];
                            const uint32_t vhi = vsm[(i * kThreads + tid) * 2 + 1];
                            float part[kRowBatch];
                            CUTLASS_PRAGMA_UNROLL
                            for (int mm = 0; mm < kRowBatch; ++mm) {
                                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                                CUTLASS_PRAGMA_UNROLL
                                for (int e = 0; e < 32; e += 4) {
                                    a0 += (vlo & (1u << e)) ? xf[mm][e] : -xf[mm][e];
                                    a1 += (vlo & (1u << (e + 1))) ? xf[mm][e + 1] : -xf[mm][e + 1];
                                    a2 += (vlo & (1u << (e + 2))) ? xf[mm][e + 2] : -xf[mm][e + 2];
                                    a3 += (vlo & (1u << (e + 3))) ? xf[mm][e + 3] : -xf[mm][e + 3];
                                }
                                CUTLASS_PRAGMA_UNROLL
                                for (int e = 0; e < 8; e += 4) {
                                    a0 += (vhi & (1u << e)) ? xf[mm][32 + e] : -xf[mm][32 + e];
                                    a1 += (vhi & (1u << (e + 1))) ? xf[mm][32 + e + 1] : -xf[mm][32 + e + 1];
                                    a2 += (vhi & (1u << (e + 2))) ? xf[mm][32 + e + 2] : -xf[mm][32 + e + 2];
                                    a3 += (vhi & (1u << (e + 3))) ? xf[mm][32 + e + 3] : -xf[mm][32 + e + 3];
                                }
                                part[mm] = (a0 + a1) + (a2 + a3);
                            }
                            CUTLASS_PRAGMA_UNROLL
                            for (int ii = 0; ii < kMaxChunk; ++ii) {
                                if (ii == i) {
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int mm = 0; mm < kRowBatch; ++mm) acc[mm][ii] += part[mm];
                                }
                            }
                        }
                    }
                    if (st != nullptr && tid == 0) {
                        if (__float_as_int(acc[0][0]) == 0x7fc00001) st[3] = -2;   // wait for compute
                        st[2] = globaltimer();
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kMaxChunk; ++i) {
                        sv[i] = __uint_as_float(uint32_t(s_raw[i]) << 16);     // bf16 -> fp32
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int mm = 0; mm < kRowBatch; ++mm) {
                        if (mm >= mrows) break;                            // warp-uniform
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kMaxChunk; ++i) {
                            if (i >= nch) break;                           // warp-uniform
                            float a = acc[mm][i];
                            CUTLASS_PRAGMA_UNROLL
                            for (int off = 16; off > 0; off >>= 1) {
                                a += __shfl_xor_sync(0xffffffffu, a, off);
                            }
                            if (lane == 0) red[warp * kRed + mm * kMaxChunk + i] = a;
                        }
                    }
                }
                if (st != nullptr && tid == 0) {
                    st[2] = globaltimer();
                }
                // publish this group: reduce the 4 warps through smem, store complete values into region 0
                asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(kThreads) : "memory");
                for (int pr = tid; pr < mrows * nch; pr += kThreads) {
                    const int mm = pr / nch;
                    const int i = pr - mm * nch;
                    const int idx = mm * kMaxChunk + i;
                    const float tot = red[idx] + red[kRed + idx] + red[2 * kRed + idx] + red[3 * kRed + idx];
                    const float svi = __uint_as_float(uint32_t(s_raw[i]) << 16);
                    __stcg(P.z_ptr + static_cast<size_t>(m0 + mm) * RankStatic + i0 + i, svi * tot);
                }
                asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(kThreads) : "memory");   // red reuse by the next group
                if (P.num_side == 0) break;                            // tile fallback: single group (host-checked)
                }   // rank groups
                if (tid == 0) {
                    asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(P.flags_ptr) : "memory");
                }
                if (st != nullptr && tid == 0) st[3] = globaltimer();
            }

            // Side unit v2 (kind 0): unit = (row pair rg, 8-rank chunk c); 128 threads, thread t owns
            // k in [40t, 40t+40). V comes pre-permuted (4 contiguous 16-byte loads per thread), x as 5
            // uint4 per row, S as one uint4. Publishes tagged (value, epoch) pairs; no fence/counter.
            CUTLASS_DEVICE static void
            z_unit2(Params const& P, int rows, int unit, int tid, float* red, int bar_id, int64_t* st = nullptr) {
                constexpr int kRB = kRowBatch;
                const int n_rc = RankStatic / 8;
                const int n_rg = (rows + kRB - 1) / kRB;
                if (unit >= n_rg * n_rc) return;
                const int rg = unit / n_rc;
                const int c = unit - rg * n_rc;
                const int m0 = rg * kRB;
                const int mrows = (rows - m0) < kRB ? (rows - m0) : kRB;
                const int warp = tid >> 5, lane = tid & 31;
                const int K = P.k_cols;
                const int kbase = tid * 40;
                // ---- loads (all issued up front)
                uint4 vq[4];
                uint4 const* vsrc = reinterpret_cast<uint4 const*>(P.vperm_ptr + (static_cast<size_t>(c) * kThreads + tid) * 64);
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < 4; ++q) vq[q] = __ldg(vsrc + q);
                uint4 xr[kRB][5];
                CUTLASS_PRAGMA_UNROLL
                for (int mm = 0; mm < kRB; ++mm) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 5; ++j) {
                        const int k = kbase + 8 * j;
                        xr[mm][j] = (mm < mrows && k < K) ? __ldg(reinterpret_cast<uint4 const*>(P.x_ptr + static_cast<size_t>(m0 + mm) * K + k)) : make_uint4(0u, 0u, 0u, 0u);
                    }
                }
                const uint4 sq = __ldg(reinterpret_cast<uint4 const*>(P.s_ptr + c * 8));
                if (st != nullptr && tid == 0) {
                    if ((vq[0].x ^ xr[0][0].x ^ sq.x) == 0x7fc00001u) st[7] = -1;
                    st[1] = globaltimer();
                }
                // ---- unpack x
                float xf[kRB][40];
                CUTLASS_PRAGMA_UNROLL
                for (int mm = 0; mm < kRB; ++mm) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 5; ++j) {
                        const uint32_t xw[4] = {xr[mm][j].x, xr[mm][j].y, xr[mm][j].z, xr[mm][j].w};
                        CUTLASS_PRAGMA_UNROLL
                        for (int q = 0; q < 4; ++q) {
                            xf[mm][8 * j + 2 * q] = __uint_as_float(xw[q] << 16);
                            xf[mm][8 * j + 2 * q + 1] = __uint_as_float(xw[q] & 0xffff0000u);
                        }
                    }
                }
                // ---- 8 ranks: V words selected from registers by a uniform compare chain
                #pragma unroll 1
                for (int i = 0; i < 8; ++i) {
                    uint32_t vlo = vq[0].x, vhi = vq[0].y;
                    if (i == 1) { vlo = vq[0].z; vhi = vq[0].w; }
                    if (i == 2) { vlo = vq[1].x; vhi = vq[1].y; }
                    if (i == 3) { vlo = vq[1].z; vhi = vq[1].w; }
                    if (i == 4) { vlo = vq[2].x; vhi = vq[2].y; }
                    if (i == 5) { vlo = vq[2].z; vhi = vq[2].w; }
                    if (i == 6) { vlo = vq[3].x; vhi = vq[3].y; }
                    if (i == 7) { vlo = vq[3].z; vhi = vq[3].w; }
                    CUTLASS_PRAGMA_UNROLL
                    for (int mm = 0; mm < kRB; ++mm) {
                        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                        CUTLASS_PRAGMA_UNROLL
                        for (int e = 0; e < 32; e += 4) {
                            a0 += (vlo & (1u << e)) ? xf[mm][e] : -xf[mm][e];
                            a1 += (vlo & (1u << (e + 1))) ? xf[mm][e + 1] : -xf[mm][e + 1];
                            a2 += (vlo & (1u << (e + 2))) ? xf[mm][e + 2] : -xf[mm][e + 2];
                            a3 += (vlo & (1u << (e + 3))) ? xf[mm][e + 3] : -xf[mm][e + 3];
                        }
                        CUTLASS_PRAGMA_UNROLL
                        for (int e = 0; e < 8; e += 4) {
                            a0 += (vhi & (1u << e)) ? xf[mm][32 + e] : -xf[mm][32 + e];
                            a1 += (vhi & (1u << (e + 1))) ? xf[mm][32 + e + 1] : -xf[mm][32 + e + 1];
                            a2 += (vhi & (1u << (e + 2))) ? xf[mm][32 + e + 2] : -xf[mm][32 + e + 2];
                            a3 += (vhi & (1u << (e + 3))) ? xf[mm][32 + e + 3] : -xf[mm][32 + e + 3];
                        }
                        float a = (a0 + a1) + (a2 + a3);
                        CUTLASS_PRAGMA_UNROLL
                        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
                        if (lane == 0) red[warp * (kRB * 8) + mm * 8 + i] = a;
                    }
                }
                if (st != nullptr && tid == 0) st[2] = globaltimer();
                asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(kThreads) : "memory");
                if (tid < mrows * 8) {
                    const int mm = tid >> 3, i = tid & 7;
                    const int idx = mm * 8 + i;
                    const float tot = red[idx] + red[kRB * 8 + idx] + red[2 * kRB * 8 + idx] + red[3 * kRB * 8 + idx];
                    const uint32_t sw[4] = {sq.x, sq.y, sq.z, sq.w};
                    const float svi = __uint_as_float((i & 1) ? (sw[i >> 1] & 0xffff0000u) : (sw[i >> 1] << 16));
                    float2* zt = reinterpret_cast<float2*>(P.z_ptr);
                    __stcg(zt + static_cast<size_t>(m0 + mm) * RankStatic + c * 8 + i, make_float2(svi * tot, __int_as_float(P.epoch)));
                }
                if (st != nullptr && tid == 0) st[3] = globaltimer();
            }

            // ---- v3 side unit helpers (CUDA cores only)
            CUTLASS_DEVICE static uint32_t smem_u32(void const* p) {
                return static_cast<uint32_t>(__cvta_generic_to_shared(p));
            }
            CUTLASS_DEVICE static void cp_async_16(void* sdst, void const* gsrc) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_u32(sdst)), "l"(gsrc) : "memory");
            }
            CUTLASS_DEVICE static void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
            CUTLASS_DEVICE static void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
            union F2U { float2 f; unsigned long long u; };
            // acc.{x,y} += s.{x,y} * z.{x,y} on the packed fp32x2 pipe (Blackwell FFMA2)
            CUTLASS_DEVICE static void ffma2(float2& acc, float s0, float s1, float2 z) {
                F2U a, b, c;
                a.f = make_float2(s0, s1); b.f = z; c.f = acc;
                asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(c.u) : "l"(a.u), "l"(b.u));
                acc = c.f;
            }
            CUTLASS_DEVICE static void pred_add(float& acc, uint32_t masked, float x) {
                asm("{\n .reg .pred p;\n setp.ne.u32 p, %1, 0;\n @p add.f32 %0, %0, %2;\n}" : "+f"(acc) : "r"(masked), "f"(x));
            }
            CUTLASS_DEVICE static void mbar_init(uint32_t mbar, uint32_t count) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar), "r"(count) : "memory");
            }
            CUTLASS_DEVICE static void mbar_expect_tx(uint32_t mbar, uint32_t bytes) {
                asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" :: "r"(mbar), "r"(bytes) : "memory");
            }
            CUTLASS_DEVICE static void bulk_g2s(uint32_t dst, void const* src, uint32_t bytes, uint32_t mbar) {
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
            }
            CUTLASS_DEVICE static void mbar_wait(uint32_t mbar, uint32_t phase) {
                asm volatile("{\n .reg .pred p;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @p bra DONE;\n bra LAB_WAIT;\n DONE:\n}"
                             :: "r"(mbar), "r"(phase) : "memory");
            }
            CUTLASS_DEVICE static bool bar_red_and(bool ok) {
                uint32_t out;
                asm volatile("{\n .reg .pred p, q;\n setp.ne.u32 p, %1, 0;\n barrier.red.and.pred q, %2, %3, p;\n selp.u32 %0, 1, 0, q;\n}"
                             : "=r"(out) : "r"(ok ? 1u : 0u), "r"(kBarW), "r"(kThreads) : "memory");
                return out != 0u;
            }
            template <int Off>
            CUTLASS_DEVICE static float lds_f32(uint32_t addr) {
                float v;
                asm("ld.shared.f32 %0, [%1+%2];" : "=f"(v) : "r"(addr), "n"(Off));
                return v;
            }

            // 8-rank block reduction: transposed butterfly (lane l ends with the 32-lane sum of rank b + rev3(l & 7))
            CUTLASS_DEVICE static void reduce8_store(float (&part)[8], int lane, int b, float* red_w) {
                const bool b0 = (lane & 1) != 0;
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    const float send = b0 ? part[i] : part[i + 4];
                    const float keep = b0 ? part[i + 4] : part[i];
                    part[i] = keep + __shfl_xor_sync(0xffffffffu, send, 1);
                }
                const bool b1 = (lane & 2) != 0;
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 2; ++i) {
                    const float send = b1 ? part[i] : part[i + 2];
                    const float keep = b1 ? part[i + 2] : part[i];
                    part[i] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
                }
                const bool b2 = (lane & 4) != 0;
                {
                    const float send = b2 ? part[0] : part[1];
                    const float keep = b2 ? part[1] : part[0];
                    part[0] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
                }
                part[0] += __shfl_xor_sync(0xffffffffu, part[0], 8);
                part[0] += __shfl_xor_sync(0xffffffffu, part[0], 16);
                if (lane < 8) {
                    const int rk = b + 4 * (lane & 1) + 2 * ((lane >> 1) & 1) + ((lane >> 2) & 1);
                    red_w[rk] = part[0];
                }
            }

            // group 0 of a side unit: nibbles 0..NL-1 through the per-thread 16-entry LUTs (one LDS + one FADD per 4 elements)
            template <int NL>
            CUTLASS_DEVICE static void
            side_group0_lut(__nv_bfloat16 const* xrow, int rc, int utid, int warp, int lane, float* lut, uint32_t const* vs,
                            float* red_w, float* red_max, float sv, int tid, int64_t* st) {
                const uint32_t lut_t = smem_u32(lut) + utid * 4;
                uint2 xr[NL];
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < NL; ++jj) xr[jj] = __ldg(reinterpret_cast<uint2 const*>(xrow + 4 * (128 * jj + utid)));
                if (st != nullptr && tid == 0) {
                    if ((xr[0].x ^ __float_as_uint(sv)) == 0x7fc00001u) st[7] = -1;
                    st[1] = globaltimer();
                }
                float lmax = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < NL; ++jj) {
                    const float x0 = __uint_as_float(xr[jj].x << 16), x1 = __uint_as_float(xr[jj].x & 0xffff0000u);
                    const float x2 = __uint_as_float(xr[jj].y << 16), x3 = __uint_as_float(xr[jj].y & 0xffff0000u);
                    lmax = fmaxf(lmax, fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3))));
                    const float a = x0 + x1, b = x0 - x1, c = x2 + x3, d = x2 - x3;
                    const float lo[4] = {-a, b, -b, a};
                    const float hi[4] = {-c, d, -d, c};
                    float* L = lut + (jj * 16) * kThreads + utid;
                    CUTLASS_PRAGMA_UNROLL
                    for (int p = 0; p < 16; ++p) L[p * kThreads] = lo[p & 3] + hi[p >> 2];
                }
                CUTLASS_PRAGMA_UNROLL
                for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, off));
                if (lane == 0) red_max[warp] = lmax;
                cp_async_wait_all();
                asm volatile("bar.sync 12, 256;" ::: "memory");
                for (int b = 0; b < rc; b += 8) {
                    float part[8];
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t w0 = vs[(b + i) * 256 + utid];
                        const uint32_t we = w0 & 0x0F0F0F0Fu, wo = (w0 >> 4) & 0x0F0F0F0Fu;
                        float acc0, acc1;
                        acc0  = lds_f32<0 * 8192>(lut_t + (__byte_perm(we, 0u, 0x4440u) << 9));
                        acc1  = lds_f32<1 * 8192>(lut_t + (__byte_perm(wo, 0u, 0x4440u) << 9));
                        acc0 += lds_f32<2 * 8192>(lut_t + (__byte_perm(we, 0u, 0x4441u) << 9));
                        acc1 += lds_f32<3 * 8192>(lut_t + (__byte_perm(wo, 0u, 0x4441u) << 9));
                        acc0 += lds_f32<4 * 8192>(lut_t + (__byte_perm(we, 0u, 0x4442u) << 9));
                        if constexpr (NL > 5) acc1 += lds_f32<5 * 8192>(lut_t + (__byte_perm(wo, 0u, 0x4442u) << 9));
                        if constexpr (NL > 6) acc0 += lds_f32<6 * 8192>(lut_t + (__byte_perm(we, 0u, 0x4443u) << 9));
                        part[i] = acc0 + acc1;
                    }
                    reduce8_store(part, lane, b, red_w);
                }
            }

            // FFMA group: nibbles NIB0..NIB0+NQ-1 (bits 4*jj of V word WORD) with PRMT-built +-1.0 selectors, no smem
            template <int NQ, int WORD, int NIB0>
            CUTLASS_DEVICE static void
            side_group_ffma(__nv_bfloat16 const* xrow, int rc, int utid, int warp, int lane, uint32_t const* vs, float* red_w, float sv, int tid, int64_t* st) {
                uint2 xr[NQ];
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < NQ; ++jj) xr[jj] = __ldg(reinterpret_cast<uint2 const*>(xrow + 4 * (128 * (NIB0 + jj) + utid)));
                if (st != nullptr && tid == 0) {
                    if ((xr[0].x ^ __float_as_uint(sv)) == 0x7fc00001u) st[7] = -1;
                    st[1] = globaltimer();
                }
                float xd[4 * NQ];
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < NQ; ++jj) {
                    xd[4 * jj + 0] = __uint_as_float(xr[jj].x << 16);
                    xd[4 * jj + 1] = __uint_as_float(xr[jj].x & 0xffff0000u);
                    xd[4 * jj + 2] = __uint_as_float(xr[jj].y << 16);
                    xd[4 * jj + 3] = __uint_as_float(xr[jj].y & 0xffff0000u);
                }
                cp_async_wait_all();
                asm volatile("bar.sync 12, 256;" ::: "memory");
                for (int b = 0; b < rc; b += 8) {
                    float part[8];
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t w1 = vs[(b + i) * 256 + WORD * 128 + utid];
                        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                        CUTLASS_PRAGMA_UNROLL
                        for (int q = 0; q < NQ; ++q) {
                            const uint32_t sp = (((w1 >> (4 * q)) & 15u) * 0x204081u) & 0x01010101u;
                            const uint32_t sg = (sp << 7) ^ 0xBFBFBFBFu;
                            a0 = fmaf(__uint_as_float(__byte_perm(sg, 0x00800000u, 0x0644u)), xd[4 * q + 0], a0);
                            a1 = fmaf(__uint_as_float(__byte_perm(sg, 0x00800000u, 0x1644u)), xd[4 * q + 1], a1);
                            a2 = fmaf(__uint_as_float(__byte_perm(sg, 0x00800000u, 0x2644u)), xd[4 * q + 2], a2);
                            a3 = fmaf(__uint_as_float(__byte_perm(sg, 0x00800000u, 0x3644u)), xd[4 * q + 3], a3);
                        }
                        part[i] = (a0 + a1) + (a2 + a3);
                    }
                    reduce8_store(part, lane, b, red_w);
                }
            }

            // Side unit v3: (row m, ranks [i0, i0+rc)) on all 256 threads of a side CTA. Thread t (0..127 within
            // its group) owns K nibbles 128*j+t; group 0 handles nibbles j < 6 through a per-thread 16-entry LUT in
            // smem (one LDS + one FADD per 4 elements), group 1 handles nibbles 6..9 with predicated adds on
            // registers (ALU pipe). V comes pre-permuted per (rank, thread) as 8 bytes, staged once by cp.async.
            // Ranks go in blocks of 8 with a transposed butterfly reduction; both groups' partials meet in smem.
            CUTLASS_DEVICE static void
            z_unit3(Params const& P, int m, int i0, int rc, int tid, char* smem, int64_t* st = nullptr) {
                float* lut = reinterpret_cast<float*>(smem);
                uint32_t const* vs = reinterpret_cast<uint32_t const*>(smem + kSideLutBytes);   // [rc][2][128] words
                float* red = reinterpret_cast<float*>(smem + kSideLutBytes + rc * 1024);
                const int group = tid >> 7, utid = tid & 127, warp = utid >> 5, lane = tid & 31;
                const int K = P.k_cols;
                {
                    char const* vsrc = reinterpret_cast<char const*>(P.vperm_ptr) + static_cast<size_t>(i0) * 1024;
                    char* vdst = reinterpret_cast<char*>(smem + kSideLutBytes);
                    for (int c = tid; c < rc * 64; c += 2 * kThreads) cp_async_16(vdst + c * 16, vsrc + c * 16);
                    cp_async_commit();
                }
                __nv_bfloat16 const* xrow = P.x_ptr + static_cast<size_t>(m) * K;
                float sv = 0.f;
                if (tid < rc) sv = __bfloat162float(P.s_ptr[i0 + tid]);
                float* red_w = red + (group * 4 + warp) * kRcMax;
                float* red_max = red + 8 * kRcMax;
                if (P.side_mode == 1) {
                    // small units: no LUT build; both groups on the FFMA path (word 0: nibbles 0..5, word 1: nibbles 6..9)
                    if (group == 0) side_group_ffma<7, 0, 0>(xrow, rc, utid, warp, lane, vs, red_w, sv, tid, st);
                    else            side_group_ffma<3, 1, 7>(xrow, rc, utid, warp, lane, vs, red_w, sv, tid, nullptr);
                } else {
                    if (group == 0) side_group0_lut<7>(xrow, rc, utid, warp, lane, lut, vs, red_w, red_max, sv, tid, st);
                    else            side_group_ffma<3, 1, 7>(xrow, rc, utid, warp, lane, vs, red_w, sv, tid, nullptr);
                }
                if (st != nullptr && tid == 0) st[2] = globaltimer();
                asm volatile("bar.sync 12, 256;" ::: "memory");
                if (tid < rc) {
                    float tot = 0.f;
                    CUTLASS_PRAGMA_UNROLL
                    for (int g = 0; g < 8; ++g) tot += red[g * kRcMax + tid];
                    float2* zt = reinterpret_cast<float2*>(P.z_ptr);
                    const float2 val = make_float2(sv * tot, __int_as_float(P.epoch));
                    const size_t off = static_cast<size_t>(m) * RankStatic + i0 + tid;
                    const size_t stride = static_cast<size_t>(P.m_rows) * RankStatic;
                    for (int cpy = 0; cpy < P.z_copies; ++cpy) __stcg(zt + cpy * stride + off, val);
                }
                if (st != nullptr && tid == 0) st[3] = globaltimer();
                asm volatile("bar.sync 12, 256;" ::: "memory");   // smem reuse by the next unit
            }

            // Mode 6 side unit: z_i = S_i * <V_i, x_m> for one rank i from dense bf16 V [R][K] (V_perm slot), all 256
            // threads (8 bf16 per 16-byte load), warp shuffle + smem reduction, published like z_unit3.
            CUTLASS_DEVICE static void
            z_unit_dense(Params const& P, int m, int i, int tid, char* smem) {
                float* red = reinterpret_cast<float*>(smem);
                const int K = P.k_cols;
                uint4 const* xv = reinterpret_cast<uint4 const*>(P.x_ptr + static_cast<size_t>(m) * K);
                uint4 const* vv = reinterpret_cast<uint4 const*>(reinterpret_cast<__nv_bfloat16 const*>(P.vperm_ptr) + static_cast<size_t>(i) * K);
                float a = 0.f;
                for (int k8 = tid; k8 < K / 8; k8 += 256) {
                    const uint4 xq = __ldg(xv + k8), vq = __ldg(vv + k8);
                    const uint32_t xs[4] = {xq.x, xq.y, xq.z, xq.w}, vs[4] = {vq.x, vq.y, vq.z, vq.w};
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        a = fmaf(__uint_as_float(xs[j] << 16), __uint_as_float(vs[j] << 16), a);
                        a = fmaf(__uint_as_float(xs[j] & 0xffff0000u), __uint_as_float(vs[j] & 0xffff0000u), a);
                    }
                }
                CUTLASS_PRAGMA_UNROLL
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if ((tid & 31) == 0) red[tid >> 5] = a;
                asm volatile("bar.sync 12, 256;" ::: "memory");
                if (tid == 0) {
                    float tot = 0.f;
                    CUTLASS_PRAGMA_UNROLL
                    for (int w = 0; w < 8; ++w) tot += red[w];
                    float2* zt = reinterpret_cast<float2*>(P.z_ptr);
                    const float2 val = make_float2(__bfloat162float(P.s_ptr[i]) * tot, __int_as_float(P.epoch));
                    const size_t off = static_cast<size_t>(m) * RankStatic + i;
                    const size_t stride = static_cast<size_t>(P.m_rows) * RankStatic;
                    for (int cpy = 0; cpy < P.z_copies; ++cpy) __stcg(zt + cpy * stride + off, val);
                }
                asm volatile("bar.sync 12, 256;" ::: "memory");   // smem reuse by the next unit
            }

            // word w of this thread's U column without a register-indexed array (runtime idx -> local memory)
            CUTLASS_DEVICE static uint32_t select_word(uint32_t const (&arr)[kRankWords], int idx) {
                uint32_t u = 0u;
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < kRankWords; ++w) u = (w == idx) ? arr[w] : u;
                return u;
            }

            // Tile side: fetch ranks [base, base+nst) of all rows (tagged pairs) into smem_z32 with one TMA bulk
            // copy per row (one copy when the block is contiguous), completion on mbarrier `mbar` (initialised by
            // the caller, phase carried across calls); every thread checks the tags of its float4s; a CTA-wide
            // and-reduction decides on a retry. MaxRows > 1: the pairs are then compacted in place to fp32 [rows][nst].
            CUTLASS_DEVICE void
            poll_stage_all(int rows, int base, int nst, uint32_t mbar, uint32_t& phase) {
                constexpr int kPer = (kMaxRows * (kStage / 2) + kThreads - 1) / kThreads;
                const int q4 = nst / 2;
                const int n4 = rows * q4;
                const int copy = tile_n_idx % params_ptr->z_copies;
                float2 const* zsrc = reinterpret_cast<float2 const*>(params_ptr->z_ptr) + static_cast<size_t>(copy) * params_ptr->m_rows * RankStatic;
                const uint32_t sdst = smem_u32(smem_z32);
                float4 const* zt = reinterpret_cast<float4 const*>(smem_z32);
                const int tag = params_ptr->epoch;
                const uint32_t bytes = static_cast<uint32_t>(rows * nst * 8);
                const bool contiguous = (nst == RankStatic) || rows == 1;
                bar_sync(kBarW);                                          // previous readers of smem_z32 are done
                int spins = 0;
                float4 v[kPer];
                if (params_ptr->poll_mode != 1) {
                    // cp.async (LSU path, not the TMA engine that the mainloop keeps busy): each thread copies its
                    // kPer float4s into smem, then checks the tags; a CTA-wide and-reduction decides on a retry.
                    char* sdst_c = reinterpret_cast<char*>(smem_z32);
                    while (true) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kPer; ++k) {
                            const int idx = thread_idx + k * kThreads;
                            if (idx < n4) {
                                const int m = idx / q4, q = idx - m * q4;
                                cp_async_16(sdst_c + static_cast<size_t>(idx) * 16, zsrc + static_cast<size_t>(m) * RankStatic + base + 2 * q);
                            }
                        }
                        cp_async_commit();
                        cp_async_wait_all();
                        bar_sync(kBarW);
                        bool ok = true;
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kPer; ++k) {
                            const int idx = thread_idx + k * kThreads;
                            if (idx < n4) {
                                v[k] = zt[idx];
                                ok = ok && (__float_as_int(v[k].y) == tag) && (__float_as_int(v[k].w) == tag);
                            }
                        }
                        if (bar_red_and(ok)) break;
                        if (++spins > (1 << 16)) { if (params_ptr->diag_ptr != nullptr) params_ptr->diag_ptr[4095] = 3; break; }
                    }
                } else
                while (true) {
                    if (thread_idx == 0) {
                        cutlass::arch::fence_view_async_shared();
                        mbar_expect_tx(mbar, bytes);
                        if (contiguous) {
                            bulk_g2s(sdst, zsrc + base, bytes, mbar);
                        } else {
                            for (int r = 0; r < rows; ++r) {
                                bulk_g2s(sdst + static_cast<uint32_t>(r * nst * 8), zsrc + static_cast<size_t>(r) * RankStatic + base, static_cast<uint32_t>(nst * 8), mbar);
                            }
                        }
                    }
                    mbar_wait(mbar, phase);
                    phase ^= 1u;
                    bool ok = true;
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < kPer; ++k) {
                        const int idx = thread_idx + k * kThreads;
                        if (idx < n4) {
                            v[k] = zt[idx];
                            ok = ok && (__float_as_int(v[k].y) == tag) && (__float_as_int(v[k].w) == tag);
                        }
                    }
                    if (bar_red_and(ok)) break;
                    if (++spins > (1 << 16)) { if (params_ptr->diag_ptr != nullptr) params_ptr->diag_ptr[4095] = 2; break; }
                }
                if constexpr (MaxRows > 1) {
                    float2* zc = reinterpret_cast<float2*>(smem_z32);    // compact fp32 [rows][kStagePad]: pair (m, q) -> float2 m*kStagePad/2 + q
                    static_assert(kMaxRows * kStagePad <= kMaxRows * kStage * 2, "padded compact rows must fit the tagged buffer");
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < kPer; ++k) {
                        const int idx = thread_idx + k * kThreads;
                        if (idx < n4) {
                            const int m = idx / q4, q = idx - m * q4;
                            zc[m * (kStagePad / 2) + q] = make_float2(v[k].x, v[k].z);
                        }
                    }
                    bar_sync(kBarW);
                }
            }

            // Tile path (num_side == 0): this tile computes one unit.
            CUTLASS_DEVICE void
            compute_z_slice(int rows) {
                z_unit(*params_ptr, rows, tile_n_idx, params_ptr->num_workers, thread_idx, smem_red, smem_vw, kBarZ);
                stamp(2);
            }

            // Protocol A (select/add side units, side_kind 0): z is one slot of tagged (value, epoch)
            // pairs; warp 0 of each tile polls 128 float4 (256 pairs) per round with backoff and stages
            // them; the other warps wait at the barrier. Protocol B (mma K-split, side_kind 1): the
            // side CTAs atomically accumulate into z[parity] and release-add a counter; thread 0 polls
            // the counter, then all threads load z once. Tile fallback (no side CTAs): protocol A.
            CUTLASS_DEVICE void
            wait_and_stage_z(int rows) {
                const bool counter_mode = true;   // tagged polling measured slower (visibility latency); both kinds use the counter
                uint32_t* zdst16 = reinterpret_cast<uint32_t*>(smem_z16);
                float2* zdst32 = reinterpret_cast<float2*>(smem_z32);
                const int n2 = rows * (RankStatic / 2);               // pairs / 2 == float4 units (tagged) or float2 units (plain)
                if (!counter_mode) {
                    float4 const* zsrc = reinterpret_cast<float4 const*>(params_ptr->z_ptr);   // tagged pairs
                    const float tag = __int_as_float(params_ptr->epoch);
                    if (thread_idx < 32) {
                        for (int base = 0; base < n2; base += 4 * 32) {
                            float4 tmp[4];
                            int spins = 0;
                            while (true) {
                                bool ok = true;
                                CUTLASS_PRAGMA_UNROLL
                                for (int j = 0; j < 4; ++j) {
                                    const int idx = base + j * 32 + thread_idx;
                                    if (idx < n2) {
                                        tmp[j] = ld_cg_volatile(zsrc + idx);
                                        ok = ok && (__float_as_int(tmp[j].y) == __float_as_int(tag)) && (__float_as_int(tmp[j].w) == __float_as_int(tag));
                                    } else {
                                        tmp[j] = make_float4(0.f, 0.f, 0.f, 0.f);
                                    }
                                }
                                if (__all_sync(0xffffffffu, ok)) break;
                                if (++spins > (1 << 20)) {
                                    if (params_ptr->diag_ptr != nullptr) { params_ptr->diag_ptr[4095] = 1; params_ptr->diag_ptr[4094] = base + thread_idx; }
                                    break;
                                }
                                __nanosleep(200);
                            }
                            CUTLASS_PRAGMA_UNROLL
                            for (int j = 0; j < 4; ++j) {
                                const int idx = base + j * 32 + thread_idx;
                                if (idx < n2) {
                                    if constexpr (MaxRows == 1) {
                                        zdst32[idx] = make_float2(tmp[j].x, tmp[j].z);
                                    } else {
                                        const __half2 hv = __floats2half2_rn(tmp[j].x, tmp[j].z);
                                        zdst16[idx] = *reinterpret_cast<uint32_t const*>(&hv);
                                    }
                                }
                            }
                        }
                    }
                    bar_sync(kBarW);
                    return;
                }
                // protocol B
                if (thread_idx == 0) {
                    const int target = params_ptr->epoch * params_ptr->n_units;
                    int32_t v;
                    int spins = 0;
                    while (true) {
                        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(params_ptr->flags_ptr) : "memory");
                        if (v >= target) break;
                        if (++spins > (1 << 22)) {
                            if (params_ptr->diag_ptr != nullptr) { params_ptr->diag_ptr[4095] = 1; params_ptr->diag_ptr[4094] = v; }
                            break;
                        }
                    }
                }
                bar_sync(kBarW);
                const int n4 = rows * (RankStatic / 4);
                const int par = (params_ptr->num_side > 0 && params_ptr->side_kind == 1) ? (params_ptr->epoch & 1) : 0;
                // kind 1: parity regions (each of float2-slot size); kind 0 / fallback: region 0, complete values
                float4 const* zsrc = reinterpret_cast<float4 const*>(params_ptr->z_ptr + static_cast<size_t>(par) * 2 * params_ptr->m_rows * RankStatic);
                uint2* zdst = reinterpret_cast<uint2*>(smem_z16);
                float4* zd32 = reinterpret_cast<float4*>(smem_z32);
                for (int base = 0; base < n4; base += 8 * kThreads) {
                    float4 tmp[8];
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 8; ++j) {
                        const int idx = base + j * kThreads + thread_idx;
                        tmp[j] = (idx < n4) ? __ldcg(zsrc + idx) : make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 8; ++j) {
                        const int idx = base + j * kThreads + thread_idx;
                        if (idx < n4) {
                            if constexpr (MaxRows == 1) {
                                zd32[idx] = tmp[j];
                            } else {
                                const __half2 h0 = __floats2half2_rn(tmp[j].x, tmp[j].y);
                                const __half2 h1 = __floats2half2_rn(tmp[j].z, tmp[j].w);
                                zdst[idx] = make_uint2(*reinterpret_cast<uint32_t const*>(&h0), *reinterpret_cast<uint32_t const*>(&h1));
                            }
                        }
                    }
                }
            }

            // Mode 4: tagged z [rows][R] (value, epoch) pairs -> fp32 z [rows][R] in smem (M = 1: smem_z32; M > 1:
            // the correction buffer smem_part, dead until the expansion writes it). Pipelined polling: a new load of
            // each pending float4 is issued every poll_gap_ns while up to kDepth loads are in flight, so a publish is
            // seen about one round trip after it becomes visible.
            CUTLASS_DEVICE void
            poll_z_tagged(int rows) {
                constexpr int kN4Max = MaxRows * RankStatic / 2;
                constexpr int kPer = (kN4Max + kThreads - 1) / kThreads;
                // ring depth: 4 when a thread polls <= 4 float4 (finer detection), 2 up to 8, 1 beyond (M 8 r512: 2-deep spilled)
                constexpr int kDepth = MaxRows == 1 ? 4 : (kPer <= kPollDeep ? 4 : (kPer > 8 ? 1 : 2));
                static_assert(kPer <= 32, "pending mask is 32 bits");
                const int n4 = rows * (RankStatic / 2);
                const int copy = tile_n_idx % params_ptr->z_copies;       // replicas spread the polls over L2 slices
                float4 const* zt4 = reinterpret_cast<float4 const*>(params_ptr->z_ptr) + static_cast<size_t>(copy) * (params_ptr->m_rows * (RankStatic / 2));
                const int tag = params_ptr->epoch;
                const int gap = kLeanTile ? 0 : params_ptr->poll_gap_ns;   // lean kernels: gap code compiled out
                // anchor: the first kDepth issues wait for tile entry + anchor (+ k gap): same absolute time on every tile
                const int anchor = params_ptr->poll_anchor_ns;
                int64_t t_hold = 0;
                if (anchor > 0) t_hold = static_cast<int64_t>((static_cast<uint64_t>(smem_vw[1]) << 32) | smem_vw[0]) + anchor;
                const int anchor_gap = params_ptr->poll_anchor_gap_ns;
                int n_issued = 0;
                auto hold = [&]() {
                    if (anchor > 0 && n_issued < kDepth) {
                        const int64_t tt = t_hold + static_cast<int64_t>(n_issued) * anchor_gap;
                        while (globaltimer() < tt) __nanosleep(64);
                    }
                    ++n_issued;
                };
                float2* zd = reinterpret_cast<float2*>((MaxRows == 1 || MaxRows <= 8) ? smem_z32 : smem_part);
                uint32_t pend = 0u;
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < kPer; ++k) {
                    const int f = thread_idx + k * kThreads;
                    if (f < n4) pend |= (1u << k);
                    else if (f < kN4Max) zd[f] = make_float2(0.f, 0.f);   // rows >= m_rows: zero z
                }
                float4 ring[kDepth][kPer];
                CUTLASS_PRAGMA_UNROLL
                for (int d = 0; d < kDepth - 1; ++d) {
                    hold();
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < kPer; ++k) {
                        if (pend & (1u << k)) ring[d][k] = ld_cg_volatile(zt4 + thread_idx + k * kThreads);
                    }
                    if (gap > 0) __nanosleep(gap);
                }
                int spins = 0;
                bool done = false;
                while (!done) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int d = 0; d < kDepth; ++d) {
                        // issue into slot (d + kDepth - 1) % kDepth, then consume slot d (the oldest)
                        const int ds = (d + kDepth - 1) % kDepth;
                        hold();
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kPer; ++k) {
                            if (pend & (1u << k)) ring[ds][k] = ld_cg_volatile(zt4 + thread_idx + k * kThreads);
                        }
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kPer; ++k) {
                            if (pend & (1u << k)) {
                                const float4 vv = ring[d][k];
                                if (__float_as_int(vv.y) == tag && __float_as_int(vv.w) == tag) {
                                    zd[thread_idx + k * kThreads] = make_float2(vv.x, vv.z);
                                    pend &= ~(1u << k);
                                }
                            }
                        }
                        if (bar_red_and(pend == 0u)) { done = true; break; }
                        if (gap > 0) __nanosleep(gap);
                    }
                    if (++spins > (1 << 20)) { if (params_ptr->diag_ptr != nullptr) params_ptr->diag_ptr[4095] = 4; break; }
                }
            }

            // Mode 4, z4b: z [copies][rows][R] as 4-byte values, 8-bit tag in the low mantissa bits (4 z per uint4 load)
            // -> fp32 z [rows][R] in smem (M = 1: smem_z32; M > 1: smem_part). Pipelined like poll_z_tagged.
            CUTLASS_DEVICE void
            poll_z_tagged4(int rows) {
                constexpr int kN4Max = MaxRows * RankStatic / 4;
                constexpr int kPer = (kN4Max + kThreads - 1) / kThreads;
                constexpr int kDepth = kPer <= 4 ? 4 : 2;
                static_assert(kPer <= 32, "pending mask is 32 bits");
                const int n4 = rows * (RankStatic / 4);
                const int copy = tile_n_idx % params_ptr->z_copies;
                uint4 const* zt = reinterpret_cast<uint4 const*>(params_ptr->z_ptr) + static_cast<size_t>(copy) * (params_ptr->m_rows * (RankStatic / 4));
                const uint32_t tag = static_cast<uint32_t>(params_ptr->epoch % 255) + 1u;
                const int gap = params_ptr->poll_gap_ns;
                float4* zd = reinterpret_cast<float4*>(MaxRows == 1 ? smem_z32 : smem_part);
                uint32_t pend = 0u;
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < kPer; ++k) {
                    const int f = thread_idx + k * kThreads;
                    if (f < n4) pend |= (1u << k);
                    else if (f < kN4Max) zd[f] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
                auto ld = [&](int k) -> uint4 {
                    uint4 v;
                    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "l"(zt + thread_idx + k * kThreads) : "memory");
                    return v;
                };
                uint4 ring[kDepth][kPer];
                CUTLASS_PRAGMA_UNROLL
                for (int d = 0; d < kDepth - 1; ++d) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < kPer; ++k) {
                        if (pend & (1u << k)) ring[d][k] = ld(k);
                    }
                    if (gap > 0) __nanosleep(gap);
                }
                int spins = 0;
                bool done = false;
                while (!done) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int d = 0; d < kDepth; ++d) {
                        const int ds = (d + kDepth - 1) % kDepth;
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kPer; ++k) {
                            if (pend & (1u << k)) ring[ds][k] = ld(k);
                        }
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kPer; ++k) {
                            if (pend & (1u << k)) {
                                const uint4 g = ring[d][k];
                                if ((g.x & 0xFFu) == tag && (g.y & 0xFFu) == tag && (g.z & 0xFFu) == tag && (g.w & 0xFFu) == tag) {
                                    zd[thread_idx + k * kThreads] = make_float4(__uint_as_float(g.x & 0xFFFFFF00u), __uint_as_float(g.y & 0xFFFFFF00u),
                                                                                __uint_as_float(g.z & 0xFFFFFF00u), __uint_as_float(g.w & 0xFFFFFF00u));
                                    pend &= ~(1u << k);
                                }
                            }
                        }
                        if (bar_red_and(pend == 0u)) { done = true; break; }
                        if (gap > 0) __nanosleep(gap);
                    }
                    if (++spins > (1 << 20)) { if (params_ptr->diag_ptr != nullptr) params_ptr->diag_ptr[4095] = 6; break; }
                }
            }

            // Mode 4, 3-4 rows in the 4-row kernel: 16-entry nibble LUT over z holding the 4 rows (float4 per entry, no signs):
            // lut [R/4][16] float4, entry p of nibble j at slot p ^ (j & 15) (conflict-free build stores and lookups without
            // padding). The LUT overlays z (tail of smem_lut4): z of a nibble is read to registers before the first barrier.
            CUTLASS_DEVICE void
            expand_lut_4rows16(int rows, uint32_t const (&uw)[kRankWords], bool help = false) {
                constexpr int kJ = RankStatic / 4;
                constexpr int kHW = kHelpA;                  // words per rank half (kHelp: the helper warps take the rest)
                static_assert(kRankWords % 2 == 0 && (8 * kHW) % 16 == 0, "rank halves on 16-nibble boundaries");
                float const* zs = smem_z32;                  // z [4][R] (rows >= m_rows zeroed by the poll), inside smem_lut4
                float4* lut = reinterpret_cast<float4*>(smem_lut4);
                constexpr int kPer = (kJ + kThreads - 1) / kThreads;
                float4 zq[kPer][4];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kPer; ++i) {
                    const int j = thread_idx + i * kThreads;
                    if (j < kJ) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int m = 0; m < 4; ++m) zq[i][m] = *reinterpret_cast<float4 const*>(zs + m * RankStatic + 4 * j);
                    }
                }
                bar_sync(kBarW);                             // all z read before the LUT overwrites it
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kPer; ++i) {
                    const int j = thread_idx + i * kThreads;
                    if (j < kJ) {
                        float e[4][16];
                        CUTLASS_PRAGMA_UNROLL
                        for (int m = 0; m < 4; ++m) bsvd_cl::lut16(zq[i][m].x, zq[i][m].y, zq[i][m].z, zq[i][m].w, e[m]);
                        CUTLASS_PRAGMA_UNROLL
                        for (int p = 0; p < 16; ++p) lut[j * 16 + (p ^ (j & 15))] = make_float4(e[0][p], e[1][p], e[2][p], e[3][p]);
                    }
                }
                bar_sync(kBarW);
                stamp(1);
                if (help) asm volatile("bar.arrive 12, 192;" ::: "memory");   // LUT ready -> helper warps
                const int c = thread_idx & 63;
                const int h = thread_idx >> 6;
                const bool ok = (tile_n_start + c) < params_ptr->n_cols;
                float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = make_float4(0.f, 0.f, 0.f, 0.f);
                float4 const* Lh = lut + h * (8 * kHW) * 16;
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < kHW; ++w) {
                    uint32_t u = h ? uw[kHW + w] : uw[w];
                    if constexpr (RankStatic >= 512) asm volatile("" : "+r"(u));   // keep the address math after the barrier
                    float4 l[8];
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < 8; ++q) {
                        const int jl = 8 * w + q;               // nibble within the half; (h * 8 kHW + jl) & 15 == jl & 15
                        l[q] = Lh[jl * 16 + static_cast<int>(((u >> (4 * q)) & 15u) ^ static_cast<uint32_t>(jl & 15))];
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < 8; q += 2) {
                        a.x += l[q].x; a.y += l[q].y; a.z += l[q].z; a.w += l[q].w;
                        b.x += l[q + 1].x; b.y += l[q + 1].y; b.z += l[q + 1].z; b.w += l[q + 1].w;
                    }
                }
                stamp(2);
                const float r0v = a.x + b.x, r1v = a.y + b.y, r2v = a.z + b.z, r3v = a.w + b.w;
                if (h == 1) {
                    smem_part[kMaxRows * kTileN + c] = r0v;
                    smem_part[(kMaxRows + 1) * kTileN + c] = r1v;
                    smem_part[(kMaxRows + 2) * kTileN + c] = r2v;
                    smem_part[(kMaxRows + 3) * kTileN + c] = r3v;
                }
                bar_sync(kBarW);
                if (help) asm volatile("bar.sync 13, 192;" ::: "memory");     // helper partials ready
                if (h == 0) {
                    const float* hp = smem_part + 2 * kMaxRows * kTileN + c;
                    const float x0 = help ? hp[0] : 0.f, x1 = help ? hp[kTileN] : 0.f, x2 = help ? hp[2 * kTileN] : 0.f, x3 = help ? hp[3 * kTileN] : 0.f;
                    smem_part[c] = ok ? r0v + smem_part[kMaxRows * kTileN + c] + x0 : 0.f;
                    smem_part[kTileN + c] = (ok && rows > 1) ? r1v + smem_part[(kMaxRows + 1) * kTileN + c] + x1 : 0.f;
                    smem_part[2 * kTileN + c] = (ok && rows > 2) ? r2v + smem_part[(kMaxRows + 2) * kTileN + c] + x2 : 0.f;
                    smem_part[3 * kTileN + c] = (ok && rows > 3) ? r3v + smem_part[(kMaxRows + 3) * kTileN + c] + x3 : 0.f;
                }
            }

            // Mode 4, M <= 2 in the 4-row kernel: corr[m][n] (m < 2) via a 16-entry nibble LUT over z holding both rows
            // (lut [R/4][16 patterns][2 rows], nibble stride 36 floats: the build's 16-byte stores hit distinct banks per
            // quarter warp); one LDS.64 + 2 FADD per 4 ranks, no signs. Thread = (column c, rank half h).
            CUTLASS_DEVICE void
            expand_lut_2rows(int rows, uint32_t const (&uw)[kRankWords], bool help = false) {
                constexpr int kJ = RankStatic / 4;
                constexpr int kJS = 36;
                static_assert(kRankWords % 2 == 0, "rank halves on word boundaries");
                constexpr int kHW = kHelpA;                  // words per rank half (kHelp: the helper warps take the rest)
                float const* zs = smem_z32;                  // z [MaxRows][R] (rows >= m_rows zeroed by the poll)
                float* lut = smem_lut4;
                for (int j = thread_idx; j < kJ; j += kThreads) {
                    const float4 za = *reinterpret_cast<float4 const*>(zs + ((4 * j) & am_));
                    const float4 zb = *reinterpret_cast<float4 const*>(zs + RankStatic + ((4 * j) & am_));
                    float ea[16], eb[16];
                    bsvd_cl::lut16(za.x, za.y, za.z, za.w, ea);
                    bsvd_cl::lut16(zb.x, zb.y, zb.z, zb.w, eb);
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < 8; ++q) {
                        if (wr_) *reinterpret_cast<float4*>(lut + j * kJS + 4 * q) = make_float4(ea[2 * q], eb[2 * q], ea[2 * q + 1], eb[2 * q + 1]);
                    }
                }
                bar_sync(kBarW);
                stamp(1);
                if (help) asm volatile("bar.arrive 12, 192;" ::: "memory");   // LUT ready -> helper warps
                const int c = thread_idx & 63;
                const int h = thread_idx >> 6;
                const bool ok = (tile_n_start + c) < params_ptr->n_cols;
                float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
                float const* Lh = lut + h * (8 * kHW) * kJS;
                const uint32_t m30 = 30u & static_cast<uint32_t>(am_);    // 2 * nibble mask (0 in the warm pass)
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < kHW; ++w) {
                    uint32_t u = h ? uw[kHW + w] : uw[w];
                    if constexpr (RankStatic >= 512) asm volatile("" : "+r"(u));   // keep the address math after the barrier
                    float2 l[8];
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < 8; ++q) {
                        const uint32_t n2 = (q == 0 ? (u << 1) : (u >> (4 * q - 1))) & m30;   // 2 * nibble q
                        l[q] = *reinterpret_cast<float2 const*>(Lh + (8 * w + q) * kJS + static_cast<int>(n2));
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < 8; q += 2) {
                        a0 += l[q].x; a1 += l[q].y;
                        b0 += l[q + 1].x; b1 += l[q + 1].y;
                    }
                }
                stamp(2);
                if (h == 1 && wr_) {
                    smem_part[kMaxRows * kTileN + c] = a0 + b0;
                    smem_part[(kMaxRows + 1) * kTileN + c] = a1 + b1;
                }
                bar_sync(kBarW);
                if (help) asm volatile("bar.sync 13, 192;" ::: "memory");     // helper partials ready
                if (h == 0 && wr_) {
                    const float x0 = help ? smem_part[2 * kMaxRows * kTileN + c] : 0.f;
                    const float x1 = help ? smem_part[(2 * kMaxRows + 1) * kTileN + c] : 0.f;
                    smem_part[c] = ok ? (a0 + b0) + smem_part[kMaxRows * kTileN + c] + x0 : 0.f;
                    smem_part[kTileN + c] = (ok && rows > 1) ? (a1 + b1) + smem_part[(kMaxRows + 1) * kTileN + c] + x1 : 0.f;
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 2; m < MaxRows; ++m) smem_part[m * kTileN + c] = 0.f;
                }
            }

            // Mode 4, M <= 2, v2 (2-row kernel): the same 16-entry nibble LUT over z (both rows per entry) in 128-byte rows,
            // 16-byte chunk p of row j at chunk p ^ (j & 7) (conflict-free build stores). Thread = (column 16 warp + (lane & 15),
            // rank half lane >> 4): the halves combine with one shuffle. The lookup byte offsets (swizzle included) are packed
            // 4 per register before the LUT barrier, so a lookup is PRMT (offset byte into the half base) + LDS.64 + 2 FADD.
            // uw: this thread's kRankWords / 2 half words (the v2 mapping, loaded in begin()).
            CUTLASS_DEVICE void
            expand_lut_2rows_v2(int rows, uint32_t const (&uw)[kRankWords]) {
                constexpr int kJ = RankStatic / 4;
                constexpr int kHW = kRankWords / 2;          // words per rank half
                constexpr int kJh = kJ / 2;                  // LUT rows per half
                static_assert(kRankWords % 2 == 0 && kJh % 8 == 0, "rank halves on word boundaries");
                float const* zs = smem_z32;                  // z [MaxRows][R] (rows >= m_rows zeroed by the poll)
                float* lut = smem_lut4;
                const int lane = thread_idx & 31;
                const int c = (thread_idx >> 5) * 16 + (lane & 15);
                const int h = lane >> 4;
                const bool ok = (tile_n_start + c) < params_ptr->n_cols;
                // byte b of pk[2w] / pk[2w + 1]: byte offset in its LUT row of nibble 2b / 2b + 1 of word w, (n ^ 2q) * 8
                uint32_t pk[2 * kHW];
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < kHW; ++w) {
                    uint32_t u = uw[w] & static_cast<uint32_t>(am_);
                    asm volatile("" : "+r"(u));              // after the poll (the U words may still be in flight before it)
                    pk[2 * w] = ((u << 3) & 0x78787878u) ^ 0x60402000u;
                    pk[2 * w + 1] = ((u >> 1) & 0x78787878u) ^ 0x70503010u;
                }
                for (int j = thread_idx; j < kJ; j += kThreads) {
                    const float4 za = *reinterpret_cast<float4 const*>(zs + ((4 * j) & am_));
                    const float4 zb = *reinterpret_cast<float4 const*>(zs + RankStatic + ((4 * j) & am_));
                    float ea[16], eb[16];
                    bsvd_cl::lut16(za.x, za.y, za.z, za.w, ea);
                    bsvd_cl::lut16(zb.x, zb.y, zb.z, zb.w, eb);
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < 8; ++q) {
                        if (wr_) *reinterpret_cast<float4*>(lut + j * 32 + 4 * (q ^ (j & 7))) = make_float4(ea[2 * q], eb[2 * q], ea[2 * q + 1], eb[2 * q + 1]);
                    }
                }
                bar_sync(kBarW);
                stamp(1);
                const uint32_t hb = static_cast<uint32_t>(h) * static_cast<uint32_t>(kJh * 128);   // half base: byte 0 is 0
                char const* lb = reinterpret_cast<char const*>(lut);
                float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < kHW; ++w) {
                    float2 l[8];
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < 8; ++q) {
                        const uint32_t off = __byte_perm(pk[2 * w + (q & 1)], hb, 0x7650u | static_cast<uint32_t>(q >> 1));
                        l[q] = *reinterpret_cast<float2 const*>(lb + off + (8 * w + q) * 128);
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < 8; q += 2) {
                        a0 += l[q].x; a1 += l[q].y;
                        b0 += l[q + 1].x; b1 += l[q + 1].y;
                    }
                }
                stamp(2);
                float s0 = a0 + b0, s1 = a1 + b1;
                s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
                s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
                if (h == 0 && wr_) {
                    smem_part[c] = ok ? s0 : 0.f;
                    smem_part[kTileN + c] = (ok && rows > 1) ? s1 : 0.f;
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 2; m < MaxRows; ++m) smem_part[m * kTileN + c] = 0.f;
                }
            }

            // Mode 4, 2 <= M <= kRowsT (4 or 8): corr[m][n] = sum_i z[m][i] u[i][n] with a kRowsT-row sign-symmetric LUT over
            // z (smem_z32 [rows][R], from the poll), built 32 nibbles (128 ranks) at a time in smem_lut4; thread =
            // (column c, rank half h of each pass); halves combined through smem_part.
            template <int kRowsT>
            CUTLASS_DEVICE void
            expand_lut_rowsK(int rows, uint32_t const (&uw)[kRankWords]) {
                using RC = bsvd_br::RowsCfg<kRowsT>;
                constexpr int kPassN = kRowsT == 4 ? RankStatic / 4 : 32;   // 4 rows: the whole LUT in one pass (18 KB at r512)
                constexpr int kJ = RankStatic / 4;
                constexpr int kPasses = (kJ + kPassN - 1) / kPassN;
                float const* zs = smem_z32;
                float* lut = smem_lut4;
                const int c = thread_idx & 63;
                const int h = thread_idx >> 6;
                const bool ok = (tile_n_start + c) < params_ptr->n_cols;
                float acc[kRowsT];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kRowsT; ++i) acc[i] = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int ps = 0; ps < kPasses; ++ps) {
                    if (ps > 0) bar_sync(kBarW);
                    for (int t = thread_idx; t < kPassN * RC::kQ4; t += kThreads) {
                        const int jl = t % kPassN, q4 = t / kPassN;
                        const int jg = ps * kPassN + jl;
                        if (jg < kJ) {
                            float za[4][4];
                            CUTLASS_PRAGMA_UNROLL
                            for (int b = 0; b < 4; ++b) {
                                const float4 z4 = *reinterpret_cast<float4 const*>(zs + (4 * q4 + b) * RankStatic + 4 * jg);
                                za[b][0] = z4.x; za[b][1] = z4.y; za[b][2] = z4.z; za[b][3] = z4.w;
                            }
                            float e[8][4];
                            bsvd_br::lut8_rows4(za, e);
                            CUTLASS_PRAGMA_UNROLL
                            for (int p7 = 0; p7 < 8; ++p7) {
                                *reinterpret_cast<float4*>(lut + jl * RC::kJSr + p7 * RC::kPSr + 4 * q4) = make_float4(e[p7][0], e[p7][1], e[p7][2], e[p7][3]);
                            }
                        }
                    }
                    bar_sync(kBarW);
                    stamp(1);
                    // lookups in batches of 8: offsets / signs, then the loads, then the FMAs
                    constexpr int kB = (kPassN / 2) >= 8 ? 8 : (kPassN / 2);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i0 = 0; i0 < kPassN / 2; i0 += kB) {
                        int off[kB];
                        float sg[kB];
                        static_assert((kPassN / 2) % 8 == 0, "rank halves on word boundaries");
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kB; ++k) {
                            const int jl = h * (kPassN / 2) + i0 + k;
                            const int jg = ps * kPassN + jl;
                            const int w0 = (ps * kPassN + i0 + k) >> 3;             // static word index of half 0
                            constexpr int kWh = kPassN / 16;                         // words per half pass
                            uint32_t uwv = h ? ((w0 + kWh < kRankWords) ? uw[(w0 + kWh < kRankWords) ? w0 + kWh : 0] : 0u) : uw[w0];
                            // opaque: keeps the offset / sign math after the LUT barrier (hoisted above the z poll, the
                            // 64 offsets + signs of r512 spilled and serialized the lookups)
                            if constexpr (RankStatic >= 512) asm volatile("" : "+r"(uwv));
                            const uint32_t pn = (jg < kJ) ? ((uwv >> (4 * ((i0 + k) & 7))) & 15u) : 8u;
                            const uint32_t tt = (pn >> 3) - 1u;
                            off[k] = jl * RC::kJSr + static_cast<int>((pn ^ tt) & 7u) * RC::kPSr;
                            sg[k] = (jg < kJ) ? __int_as_float(0x3f800000 ^ (tt & 0x80000000u)) : 0.f;
                        }
                        float4 l[kB][RC::kQ4];
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kB; ++k) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int q4 = 0; q4 < RC::kQ4; ++q4) l[k][q4] = reinterpret_cast<float4 const*>(lut + off[k])[q4];
                        }
                        CUTLASS_PRAGMA_UNROLL
                        for (int k = 0; k < kB; ++k) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int q4 = 0; q4 < RC::kQ4; ++q4) {
                                acc[4 * q4 + 0] = fmaf(sg[k], l[k][q4].x, acc[4 * q4 + 0]);
                                acc[4 * q4 + 1] = fmaf(sg[k], l[k][q4].y, acc[4 * q4 + 1]);
                                acc[4 * q4 + 2] = fmaf(sg[k], l[k][q4].z, acc[4 * q4 + 2]);
                                acc[4 * q4 + 3] = fmaf(sg[k], l[k][q4].w, acc[4 * q4 + 3]);
                            }
                        }
                    }
                }
                stamp(2);
                if (h == 1) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 0; m < kRowsT; ++m) smem_part[(kMaxRows + m) * kTileN + c] = acc[m];
                }
                bar_sync(kBarW);
                if (h == 0) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 0; m < kRowsT; ++m) {
                        smem_part[m * kTileN + c] = (ok && m < rows) ? acc[m] + smem_part[(kMaxRows + m) * kTileN + c] : 0.f;
                    }
                }
            }

            // Mode 4, M > 1 (z path), variant 2: thread = (column c, row half rh = rows 8rh..8rh+7) over all R/4 rank
            // nibbles; same sign-symmetric LUT; each thread writes its 8 final rows (no half-combine, one barrier less).
            CUTLASS_DEVICE void
            expand_lut_rows2(int rows, uint32_t const (&uw)[kRankWords]) {
                constexpr int kJ = RankStatic / 4;
                constexpr int kPS = 20;
                constexpr int kJS = 8 * kPS;
                float const* zs = smem_part;
                float* lut = smem_lut4;
                for (int t = thread_idx; t < kJ * 4; t += kThreads) {
                    const int j = t >> 2, mq = t & 3;
                    float e[8][4];
                    CUTLASS_PRAGMA_UNROLL
                    for (int b = 0; b < 4; ++b) {
                        const float4 zq = *reinterpret_cast<float4 const*>(zs + (4 * mq + b) * RankStatic + 4 * j);
                        const float base = zq.w - zq.x - zq.y - zq.z;
                        const float d0 = 2.f * zq.x, d1 = 2.f * zq.y, d2 = 2.f * zq.z;
                        e[0][b] = base;       e[1][b] = base + d0;
                        e[2][b] = base + d1;  e[3][b] = e[1][b] + d1;
                        e[4][b] = base + d2;  e[5][b] = e[1][b] + d2;
                        e[6][b] = e[2][b] + d2; e[7][b] = e[3][b] + d2;
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int p7 = 0; p7 < 8; ++p7) {
                        *reinterpret_cast<float4*>(lut + j * kJS + p7 * kPS + 4 * mq) = make_float4(e[p7][0], e[p7][1], e[p7][2], e[p7][3]);
                    }
                }
                bar_sync(kBarW);
                stamp(1);
                const int c = thread_idx & 63;
                const int rh = thread_idx >> 6;
                const bool ok = (tile_n_start + c) < params_ptr->n_cols;
                auto pk2 = [](float a, float b) -> unsigned long long {
                    return static_cast<unsigned long long>(__float_as_uint(a)) | (static_cast<unsigned long long>(__float_as_uint(b)) << 32);
                };
                unsigned long long acc2[4] = {0ull, 0ull, 0ull, 0ull};
                auto off_of = [&](int j, float& sg) -> int {
                    const uint32_t pn = (uw[j >> 3] >> (4 * (j & 7))) & 15u;
                    const uint32_t t = (pn >> 3) - 1u;
                    const uint32_t p7 = (pn ^ t) & 7u;
                    sg = __int_as_float(0x3f800000 ^ (t & 0x80000000u));
                    return j * kJS + static_cast<int>(p7) * kPS + 8 * rh;
                };
                float sgc, sgn;
                int oc = off_of(0, sgc);
                float4 Lc0 = *reinterpret_cast<float4 const*>(lut + oc), Lc1 = *reinterpret_cast<float4 const*>(lut + oc + 4);
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < kJ; ++j) {
                    float4 Ln0 = Lc0, Ln1 = Lc1;
                    if (j + 1 < kJ) {
                        const int on = off_of(j + 1, sgn);
                        Ln0 = *reinterpret_cast<float4 const*>(lut + on);
                        Ln1 = *reinterpret_cast<float4 const*>(lut + on + 4);
                    }
                    // packed fp32x2 FMA (FFMA2): 4 instructions per nibble for 8 rows
                    {
                        const unsigned long long s2 = pk2(sgc, sgc);
                        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(acc2[0]) : "l"(s2), "l"(pk2(Lc0.x, Lc0.y)));
                        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(acc2[1]) : "l"(s2), "l"(pk2(Lc0.z, Lc0.w)));
                        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(acc2[2]) : "l"(s2), "l"(pk2(Lc1.x, Lc1.y)));
                        asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(acc2[3]) : "l"(s2), "l"(pk2(Lc1.z, Lc1.w)));
                    }
                    Lc0 = Ln0; Lc1 = Ln1; sgc = sgn;
                }
                stamp(2);
                float acc[8];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    acc[2 * i] = __uint_as_float(static_cast<uint32_t>(acc2[i]));
                    acc[2 * i + 1] = __uint_as_float(static_cast<uint32_t>(acc2[i] >> 32));
                }
                // z in smem_part was consumed by the LUT build before the first barrier: write the rows directly
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 8; ++i) {
                    const int m = 8 * rh + i;
                    smem_part[m * kTileN + c] = (ok && m < rows) ? acc[i] : 0.f;
                }
            }

            // Mode 4, M > 1, side_corr: the side units publish partial corrections per 32-rank block, 4-byte values
            // with an 8-bit tag in the low mantissa bits, [tile][block][row 16][64 cols]. Thread = 4 columns of two
            // rows (item k: block k/2, row half k&1); poll until every tag matches, sum the blocks -> smem_part.
            CUTLASS_DEVICE void
            poll_corr_rows(int rows) {
                constexpr int kBlk = RankStatic / 32;
                constexpr int kPer = 2 * kBlk;                      // uint4 per thread (tile block = kBlk * 1024 values)
                static_assert(kPer <= 32 && MaxRows == 16, "side_corr: 16-row tiles, R <= 512");
                uint4 const* src = reinterpret_cast<uint4 const*>(params_ptr->z_ptr) + static_cast<size_t>(tile_n_idx) * kBlk * 256;
                const uint32_t tag = static_cast<uint32_t>(params_ptr->epoch % 255) + 1u;
                const int gap = params_ptr->poll_gap_ns;
                int mrow[2];
                mrow[0] = (4 * thread_idx) >> 6;
                mrow[1] = (4 * thread_idx + 512) >> 6;
                const int c = (4 * thread_idx) & 63;
                uint32_t pend = 0u;
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < kPer; ++k) {
                    if (mrow[k & 1] < rows) pend |= (1u << k);
                }
                float4 acc[2] = {make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)};
                int spins = 0;
                while (true) {
                    uint4 got[kPer];
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < kPer; ++k) {
                        if (pend & (1u << k)) {
                            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(got[k].x), "=r"(got[k].y), "=r"(got[k].z), "=r"(got[k].w)
                                         : "l"(src + thread_idx + 128 * k) : "memory");
                        }
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < kPer; ++k) {
                        if (pend & (1u << k)) {
                            const uint4 g = got[k];
                            if ((g.x & 0xFFu) == tag && (g.y & 0xFFu) == tag && (g.z & 0xFFu) == tag && (g.w & 0xFFu) == tag) {
                                float4& a = acc[k & 1];
                                a.x += __uint_as_float(g.x & 0xFFFFFF00u); a.y += __uint_as_float(g.y & 0xFFFFFF00u);
                                a.z += __uint_as_float(g.z & 0xFFFFFF00u); a.w += __uint_as_float(g.w & 0xFFFFFF00u);
                                pend &= ~(1u << k);
                            }
                        }
                    }
                    if (bar_red_and(pend == 0u)) break;
                    if (gap > 0) __nanosleep(gap);
                    if (++spins > (1 << 20)) { if (params_ptr->diag_ptr != nullptr) params_ptr->diag_ptr[4095] = 5; break; }
                }
                const bool ok = (tile_n_start + c) < params_ptr->n_cols;
                CUTLASS_PRAGMA_UNROLL
                for (int b = 0; b < 2; ++b) {
                    const float4 v = (ok && mrow[b] < rows) ? acc[b] : make_float4(0.f, 0.f, 0.f, 0.f);
                    *reinterpret_cast<float4*>(smem_part + mrow[b] * kTileN + c) = v;
                }
            }

            // Mode 4, M = 1: corr[n] = sum_i z[i] u[i][n] via a 16-entry nibble LUT over z (smem_lut4 [R/4][16]):
            // one LDS + one FADD per 4 ranks. Thread = (column c, rank half h) as in expand_fsel_row0.
            CUTLASS_DEVICE void
            expand_lut_row0(uint32_t const (&uw)[(kRankWords + 1) / 2]) {
                constexpr int kJ = RankStatic / 4;
                float* lut = smem_lut4;
                float4 const* z4 = reinterpret_cast<float4 const*>(smem_z32);
                for (int j = thread_idx; j < kJ; j += kThreads) {
                    const float4 zq = z4[j];
                    float e[16];
                    bsvd_cl::lut16(zq.x, zq.y, zq.z, zq.w, e);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        reinterpret_cast<float4*>(lut + 16 * j)[i] = make_float4(e[4 * i], e[4 * i + 1], e[4 * i + 2], e[4 * i + 3]);
                    }
                }
                bar_sync(kBarW);
                stamp(1);
                const int c = thread_idx & 63;
                const int h = thread_idx >> 6;
                const bool ok = (tile_n_start + c) < params_ptr->n_cols;
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < kHalfWords; ++w) {
                    const int ww = h * kHalfWords + w;
                    if (ww * 32 >= RankStatic) break;
                    const uint32_t u = uw[w];
                    float const* L = lut + ww * 8 * 16;
                    CUTLASS_PRAGMA_UNROLL
                    for (int q8 = 0; q8 < 8; ++q8) {
                        if (ww * 32 + 4 * q8 >= RankStatic) break;
                        acc[q8 & 3] += L[q8 * 16 + static_cast<int>((u >> (4 * q8)) & 15u)];
                    }
                }
                stamp(2);
                const float part = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                smem_part[h * kTileN + c] = ok ? part : 0.f;
                bar_sync(kBarW);
                if (h == 1) {
                    smem_part[c] += smem_part[kTileN + c];
                }
            }

            // Mode 4, M > 1: corr[m][n] = sum_i z[m][i] u[i][n] on CUDA cores via nibble LUTs over z.
            // LUT [R/4 nibbles j][8 patterns][16 rows + 4 pad] holds, for U nibble p = 8 | p7 (rank 4j+3 bit set),
            // sum_i s_i z[m][4j+i]; a nibble without bit 3 is the negation of the entry at (~p) & 7. Thread =
            // (column c, rank half h): per nibble one PRMT-free index + 4 LDS.128 (16 rows) + 16 FFMA (sign).
            // The 80-byte pattern stride puts the 8 patterns of a nibble in 8 distinct bank groups.
            CUTLASS_DEVICE void
            expand_lut_rows(int rows, uint32_t const (&uw)[(kRankWords + 1) / 2]) {
                constexpr int kJ = RankStatic / 4;
                constexpr int kPS = 20;                      // floats per pattern
                constexpr int kJS = 8 * kPS;                 // floats per nibble
                float const* zs = smem_part;                 // z [16][R] fp32 (poll_z_tagged)
                float* lut = smem_lut4;
                // build: (nibble j, row quad mq) per task
                for (int t = thread_idx; t < kJ * 4; t += kThreads) {
                    const int j = t >> 2, mq = t & 3;
                    float e[8][4];
                    CUTLASS_PRAGMA_UNROLL
                    for (int b = 0; b < 4; ++b) {
                        const float4 zq = *reinterpret_cast<float4 const*>(zs + (4 * mq + b) * RankStatic + 4 * j);
                        const float base = zq.w - zq.x - zq.y - zq.z;
                        const float d0 = 2.f * zq.x, d1 = 2.f * zq.y, d2 = 2.f * zq.z;
                        e[0][b] = base;       e[1][b] = base + d0;
                        e[2][b] = base + d1;  e[3][b] = e[1][b] + d1;
                        e[4][b] = base + d2;  e[5][b] = e[1][b] + d2;
                        e[6][b] = e[2][b] + d2; e[7][b] = e[3][b] + d2;
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int p7 = 0; p7 < 8; ++p7) {
                        *reinterpret_cast<float4*>(lut + j * kJS + p7 * kPS + 4 * mq) = make_float4(e[p7][0], e[p7][1], e[p7][2], e[p7][3]);
                    }
                }
                bar_sync(kBarW);
                stamp(1);
                const int c = thread_idx & 63;
                const int h = thread_idx >> 6;
                const bool ok = (tile_n_start + c) < params_ptr->n_cols;
                float acc[16];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 16; ++i) acc[i] = 0.f;
                // per nibble of this half: LUT row offset and sign (all ALU work first), then the lookups
                // software-pipelined one nibble ahead so LDS latency overlaps the FMAs
                constexpr int kJh = kHalfWords * 8;
                int offs[kJh];
                float sgs[kJh];
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < kJh; ++jj) {
                    const int w = jj >> 3, q8 = jj & 7;
                    const int ww = h * kHalfWords + w;
                    const bool valid = ww * 32 + 4 * q8 < RankStatic;
                    const uint32_t pn = (uw[w] >> (4 * q8)) & 15u;
                    const uint32_t t = (pn >> 3) - 1u;                           // 0 if bit 3 set, else all ones
                    const uint32_t p7 = (pn ^ t) & 7u;
                    offs[jj] = valid ? (ww * 8 + q8) * kJS + static_cast<int>(p7) * kPS : 0;
                    sgs[jj] = valid ? __int_as_float(0x3f800000 ^ (t & 0x80000000u)) : 0.f;
                }
                float4 Lc[4], Ln[4];
                CUTLASS_PRAGMA_UNROLL
                for (int qq = 0; qq < 4; ++qq) Lc[qq] = reinterpret_cast<float4 const*>(lut + offs[0])[qq];
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < kJh; ++jj) {
                    if (jj + 1 < kJh) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int qq = 0; qq < 4; ++qq) Ln[qq] = reinterpret_cast<float4 const*>(lut + offs[jj + 1])[qq];
                    }
                    const float sg = sgs[jj];
                    CUTLASS_PRAGMA_UNROLL
                    for (int qq = 0; qq < 4; ++qq) {
                        acc[4 * qq + 0] = fmaf(sg, Lc[qq].x, acc[4 * qq + 0]);
                        acc[4 * qq + 1] = fmaf(sg, Lc[qq].y, acc[4 * qq + 1]);
                        acc[4 * qq + 2] = fmaf(sg, Lc[qq].z, acc[4 * qq + 2]);
                        acc[4 * qq + 3] = fmaf(sg, Lc[qq].w, acc[4 * qq + 3]);
                    }
                    if (jj + 1 < kJh) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int qq = 0; qq < 4; ++qq) Lc[qq] = Ln[qq];
                    }
                }
                stamp(2);
                // half 1 -> smem_part[1], half 0 adds and writes the final rows (z in smem_part is dead after the build bar)
                if (h == 1) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 0; m < 16; ++m) {
                        if (m < MaxRows) smem_part[(kMaxRows + m) * kTileN + c] = acc[m];
                    }
                }
                bar_sync(kBarW);
                if (h == 0) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 0; m < 16; ++m) {
                        if (m < MaxRows) smem_part[m * kTileN + c] = (ok && m < rows) ? acc[m] + smem_part[(kMaxRows + m) * kTileN + c] : 0.f;
                    }
                }
            }

            // External fp32 z -> smem (fp16 for mma, fp32 for the M=1 path), batched loads.
            CUTLASS_DEVICE void
            stage_external_z(int rows) {
                const int n4 = rows * (RankStatic / 4);
                float4 const* zsrc = reinterpret_cast<float4 const*>(
                    params_ptr->z_ptr + static_cast<size_t>(tile_m_start) * RankStatic);
                uint2* zdst = reinterpret_cast<uint2*>(smem_z16);
                float4* zdst32 = reinterpret_cast<float4*>(smem_z32);
                for (int base = 0; base < n4; base += 8 * kThreads) {
                    float4 tmp[8];
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 8; ++j) {
                        const int idx = base + j * kThreads + thread_idx;
                        tmp[j] = (idx < n4) ? __ldcg(zsrc + idx) : make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 8; ++j) {
                        const int idx = base + j * kThreads + thread_idx;
                        if (idx < n4) {
                            if constexpr (MaxRows == 1) {
                                zdst32[idx] = tmp[j];
                            } else {
                                const __half2 h0 = __floats2half2_rn(tmp[j].x, tmp[j].y);
                                const __half2 h1 = __floats2half2_rn(tmp[j].z, tmp[j].w);
                                zdst[idx] = make_uint2(*reinterpret_cast<uint32_t const*>(&h0), *reinterpret_cast<uint32_t const*>(&h1));
                            }
                        }
                    }
                }
            }

            // M=1 expansion on CUDA cores from fp32 z in smem (broadcast LDS.128):
            // thread = (column c, rank-word half h); corr = sum_i (u ? z : -z). Half 1 adds into half 0.
            static constexpr int kHalfWords = (kRankWords + 1) / 2;
            CUTLASS_DEVICE void
            load_uw_row0(uint32_t (&uw)[kHalfWords]) {
                const int c = thread_idx & 63;
                const int h = thread_idx >> 6;
                const int n = tile_n_start + c;
                const bool ok = n < params_ptr->n_cols;
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < kHalfWords; ++w) {
                    const int ww = h * kHalfWords + w;
                    uw[w] = (ok && ww < kRankWords) ? bsvd_cl::ldg_u32_early(params_ptr->u_col32_ptr + static_cast<size_t>(n) * kRankWords + ww) : 0u;
                }
            }
            CUTLASS_DEVICE void
            expand_fsel_row0(uint32_t const (&uw)[kHalfWords]) {
                const int c = thread_idx & 63;
                const int h = thread_idx >> 6;
                const int n = tile_n_start + c;
                const bool ok = n < params_ptr->n_cols;
                float4 const* z4 = reinterpret_cast<float4 const*>(smem_z32);
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll 1
                for (int w = 0; w < kHalfWords; ++w) {
                    const int ww = h * kHalfWords + w;
                    if (ww * 32 >= RankStatic) break;
                    const uint32_t u = uw[w];
                    if (ww * 32 + 32 <= RankStatic) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int q = 0; q < 8; ++q) {
                            const float4 zv = z4[ww * 8 + q];
                            const uint32_t nib = u >> (4 * q);
                            acc[0] += (nib & 1u) ? zv.x : -zv.x;
                            acc[1] += (nib & 2u) ? zv.y : -zv.y;
                            acc[2] += (nib & 4u) ? zv.z : -zv.z;
                            acc[3] += (nib & 8u) ? zv.w : -zv.w;
                        }
                    } else {
                        CUTLASS_PRAGMA_UNROLL
                        for (int q = 0; q < 4; ++q) {                      // rank 16: half a word
                            const float4 zv = z4[ww * 8 + q];
                            const uint32_t nib = u >> (4 * q);
                            acc[0] += (nib & 1u) ? zv.x : -zv.x;
                            acc[1] += (nib & 2u) ? zv.y : -zv.y;
                            acc[2] += (nib & 4u) ? zv.z : -zv.z;
                            acc[3] += (nib & 8u) ? zv.w : -zv.w;
                        }
                    }
                }
                const float part = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                smem_part[h * kTileN + c] = ok ? part : 0.f;
                bar_sync(kBarW);
                if (h == 1) {
                    smem_part[c] += smem_part[kTileN + c];
                }
            }

            // corr[m][n] = sum_i z[m][i] u[i][n] on tensor cores (f16 in, fp32 acc). Warp w owns
            // columns [16w, 16w+16) = two n8 tiles; k-step s covers ranks [16s, 16s+16), permuted so
            // lane t reads ranks 16s+4t+{0..3} (one 8-byte smem load per row, one nibble of U bits).
            CUTLASS_DEVICE void
            load_uw_mma(uint32_t (&uw0)[kRankWords], uint32_t (&uw1)[kRankWords]) {
                const int warp = thread_idx >> 5;
                const int lane = thread_idx & 31;
                const int g = lane >> 2;
                const int c0 = tile_n_start + 16 * warp + g;
                const int c1 = c0 + 8;
                const bool ok0 = c0 < params_ptr->n_cols;
                const bool ok1 = c1 < params_ptr->n_cols;
                CUTLASS_PRAGMA_UNROLL
                for (int w = 0; w < kRankWords; ++w) {
                    uw0[w] = ok0 ? static_cast<uint32_t>(__ldg(params_ptr->u_col32_ptr + static_cast<size_t>(c0) * kRankWords + w)) : 0u;
                    uw1[w] = ok1 ? static_cast<uint32_t>(__ldg(params_ptr->u_col32_ptr + static_cast<size_t>(c1) * kRankWords + w)) : 0u;
                }
            }
            CUTLASS_DEVICE void
            expand_mma(int rows, uint32_t const (&uw0)[kRankWords], uint32_t const (&uw1)[kRankWords]) {
                const int warp = thread_idx >> 5;
                const int lane = thread_idx & 31;
                const int g = lane >> 2;
                const int t = lane & 3;
                const bool row_lo = g < rows;
                const bool row_hi = (g + 8) < rows;
                const int c0 = tile_n_start + 16 * warp + g;
                const int c1 = c0 + 8;
                const bool ok0 = c0 < params_ptr->n_cols;
                const bool ok1 = c1 < params_ptr->n_cols;
                float e0[4][4], e1[4][4];
                CUTLASS_PRAGMA_UNROLL
                for (int ch = 0; ch < 4; ++ch) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 4; ++j) { e0[ch][j] = 0.f; e1[ch][j] = 0.f; }
                }
                uint32_t const* zlo = reinterpret_cast<uint32_t const*>(smem_z16 + g * RankStatic);
                uint32_t const* zhi = reinterpret_cast<uint32_t const*>(smem_z16 + (g + 8) * RankStatic);
                CUTLASS_PRAGMA_UNROLL
                for (int sidx = 0; sidx < RankStatic / 16; ++sidx) {
                    const int r0 = 16 * sidx + 4 * t;
                    uint2 al = row_lo ? *reinterpret_cast<uint2 const*>(zlo + (r0 >> 1)) : make_uint2(0u, 0u);
                    uint2 ah = row_hi ? *reinterpret_cast<uint2 const*>(zhi + (r0 >> 1)) : make_uint2(0u, 0u);
                    const uint32_t a[4] = {al.x, ah.x, al.y, ah.y};
                    const uint32_t n0 = (uw0[r0 >> 5] >> (r0 & 31)) & 15u;
                    const uint32_t n1 = (uw1[r0 >> 5] >> (r0 & 31)) & 15u;
                    const uint32_t b0[2] = {sign_pair_f16(n0 & 1u, (n0 >> 1) & 1u), sign_pair_f16((n0 >> 2) & 1u, (n0 >> 3) & 1u)};
                    const uint32_t b1[2] = {sign_pair_f16(n1 & 1u, (n1 >> 1) & 1u), sign_pair_f16((n1 >> 2) & 1u, (n1 >> 3) & 1u)};
                    mma_f16_16816(e0[sidx & 3], a, b0);
                    mma_f16_16816(e1[sidx & 3], a, b1);
                }
                float d0[4] = {0.f, 0.f, 0.f, 0.f};
                float d1[4] = {0.f, 0.f, 0.f, 0.f};
                CUTLASS_PRAGMA_UNROLL
                for (int ch = 0; ch < 4; ++ch) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 4; ++j) { d0[j] += e0[ch][j]; d1[j] += e1[ch][j]; }
                }
                const int cl = 16 * warp + 2 * t;
                if (row_lo) {
                    smem_part[g * kTileN + cl] = ok0 ? d0[0] : 0.f;
                    smem_part[g * kTileN + cl + 1] = ok0 ? d0[1] : 0.f;
                    smem_part[g * kTileN + cl + 8] = ok1 ? d1[0] : 0.f;
                    smem_part[g * kTileN + cl + 9] = ok1 ? d1[1] : 0.f;
                }
                if (row_hi) {
                    smem_part[(g + 8) * kTileN + cl] = ok0 ? d0[2] : 0.f;
                    smem_part[(g + 8) * kTileN + cl + 1] = ok0 ? d0[3] : 0.f;
                    smem_part[(g + 8) * kTileN + cl + 8] = ok1 ? d1[2] : 0.f;
                    smem_part[(g + 8) * kTileN + cl + 9] = ok1 ? d1[3] : 0.f;
                }
            }

            CUTLASS_DEVICE void
            begin() {
                const int rows = rows_valid;
                const int c = thread_idx & 63;
                const int h = thread_idx >> 6;
                const int n = tile_n_start + c;
                const bool ok = n < params_ptr->n_cols;
                if constexpr (Mode == 5) {
                    // wait until every m-tile of this tile's correction is written, then stage rows < M
                    if (thread_idx == 0) {
                        const int target = params_ptr->br.epoch * (params_ptr->br.Mp / 16);
                        int const* cp = params_ptr->br.cnt + 1 + tile_n_idx;
                        while (bsvd_br::ld_acquire(cp) < target) __nanosleep(32);
                    }
                    bar_sync(kBarW);
                    float4 const* src = reinterpret_cast<float4 const*>(params_ptr->br.corr + static_cast<size_t>(tile_n_idx) * params_ptr->br.Mp * 64);
                    float4* dst = reinterpret_cast<float4*>(smem_part);
                    for (int i = thread_idx; i < rows * 16; i += kThreads) dst[i] = __ldcg(src + i);
                    bar_sync(kBarW);
                    return;
                }
                (void)c; (void)h; (void)n; (void)ok;
                // U words for this thread's columns: issued first so their latency overlaps the z wait
                uint32_t uwr[kHalfWords];
                uint32_t uwa[kRankWords], uwb[kRankWords];
                uint32_t upend_[kRankWords];                 // cheap warm: U words in flight, copied into uwa after the poll
                uint32_t uwr_all[kRankWords];
                constexpr bool side_path = kInKernelZ;
                {
                    const int n0 = tile_n_start + (thread_idx & 63);
                    const bool ok0 = n0 < params_ptr->n_cols;
                    if (kInKernelZ && (Mode == 6 || MaxRows > 1 || (params_ptr->dbg_flags & 8))) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int w = 0; w < kRankWords; ++w) uwr_all[w] = 0u;      // M > 1: U comes by TMA into smem_vw (Mode 6: dense U below)
                    } else if (params_ptr->u_rw_ptr != nullptr) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int w = 0; w < kRankWords; ++w) {
                            uwr_all[w] = (ok0 && kInKernelZ) ? bsvd_cl::ldg_u32_early(params_ptr->u_rw_ptr + static_cast<size_t>(w) * params_ptr->n_cols + n0) : 0u;
                        }
                    } else {
                        CUTLASS_PRAGMA_UNROLL
                        for (int w = 0; w < kRankWords; ++w) {
                            uwr_all[w] = (ok0 && kInKernelZ) ? bsvd_cl::ldg_u32_early(params_ptr->u_col32_ptr + static_cast<size_t>(n0) * kRankWords + w) : 0u;
                        }
                    }
                }
                [[maybe_unused]] float ud[(Mode == 6) ? RankStatic / 2 : 1];
                if constexpr (Mode == 6) {
                    // dense BF16 U [R][N]: this thread's column, its rank half, loaded before the z wait
                    const int n0 = tile_n_start + (thread_idx & 63);
                    const int hh = thread_idx >> 6;
                    const bool ok0 = n0 < params_ptr->n_cols;
                    __nv_bfloat16 const* udp = reinterpret_cast<__nv_bfloat16 const*>(params_ptr->u_rw_ptr);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < RankStatic / 2; ++i)
                        ud[i] = ok0 ? __bfloat162float(udp[static_cast<size_t>(hh * (RankStatic / 2) + i) * params_ptr->n_cols + n0]) : 0.f;
                }
                if constexpr (MaxRows == 1) {
                    if (!side_path) load_uw_row0(uwr);
                    else {
                        CUTLASS_PRAGMA_UNROLL
                        for (int w = 0; w < kHalfWords; ++w) uwr[w] = 0u;
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int w = 0; w < kRankWords; ++w) { uwa[w] = 0u; uwb[w] = 0u; }
                } else {
                    if (!side_path && Mode != 4) load_uw_mma(uwa, uwb);
                    else {
                        CUTLASS_PRAGMA_UNROLL
                        for (int w = 0; w < kRankWords; ++w) { uwa[w] = 0u; uwb[w] = 0u; }
                    }
                    if constexpr (Mode == 4) {
                        if constexpr (MaxRows > 8) load_uw_row0(uwr);   // expand_lut_rows: thread (column, rank half), like M = 1
                        else {
                            CUTLASS_PRAGMA_UNROLL
                            for (int w = 0; w < kHalfWords; ++w) uwr[w] = 0u;   // unused by the 2- to 8-row expansions
                        }
                        if (params_ptr->exp2 || MaxRows <= 8) {
                            const int n = tile_n_start + (thread_idx & 63);
                            const bool okn = n < params_ptr->n_cols;
                            if (kV2Only || (!kLeanTile && MaxRows == 2 && params_ptr->exp2v2)) {
                                // v2 mapping: column 16 warp + (lane & 15), rank half lane >> 4; this half's words only
                                constexpr int kHW2 = kRankWords / 2;
                                const int n2 = tile_n_start + (thread_idx >> 5) * 16 + (thread_idx & 15);
                                const int h2 = (thread_idx >> 4) & 1;
                                const bool ok2 = n2 < params_ptr->n_cols;
                                auto const* src = params_ptr->u_col32_ptr + static_cast<size_t>(n2) * kRankWords + h2 * kHW2;
                                CUTLASS_PRAGMA_UNROLL
                                for (int w = 0; w < kRankWords; ++w) upend_[w] = 0u;
                                if constexpr (kHW2 % 4 == 0) {
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int w = 0; w < kHW2; w += 4) {
                                        const uint4 v4 = ok2 ? bsvd_cl::ldg_v4_early(src + w) : make_uint4(0u, 0u, 0u, 0u);
                                        upend_[w] = v4.x; upend_[w + 1] = v4.y; upend_[w + 2] = v4.z; upend_[w + 3] = v4.w;
                                    }
                                } else {
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int w = 0; w < kHW2; ++w) upend_[w] = ok2 ? bsvd_cl::ldg_u32_early(src + w) : 0u;
                                }
                            } else if ((MaxRows == 2 && params_ptr->warm_cheap) || (MaxRows == 8 && RankStatic <= 128)) {
                                // staged: the warm pass (before the poll) runs on zero U words instead of waiting ~1-2 us for
                                // these loads behind the weight stream (8-row kernel: that wait decided when the first z poll
                                // left; same binary M8 r128 7.00 -> 6.90 us)
                                CUTLASS_PRAGMA_UNROLL
                                for (int w = 0; w < kRankWords; ++w) upend_[w] = okn ? bsvd_cl::ldg_u32_early(params_ptr->u_col32_ptr + static_cast<size_t>(n) * kRankWords + w) : 0u;
                            } else {
                                CUTLASS_PRAGMA_UNROLL
                                for (int w = 0; w < kRankWords; ++w) uwa[w] = okn ? bsvd_cl::ldg_u32_early(params_ptr->u_col32_ptr + static_cast<size_t>(n) * kRankWords + w) : 0u;
                            }
                        }
                    } else {
                        CUTLASS_PRAGMA_UNROLL
                        for (int w = 0; w < kHalfWords; ++w) uwr[w] = 0u;
                    }
                }
                if constexpr (kInKernelZ) {
                    stamp(0);
                    if (side_path) {
                        const uint32_t mbar0 = smem_u32(smem_red);
                        const uint32_t mbar1 = mbar0 + 8u;
                        uint32_t zphase = 0u;
                        if ((params_ptr->dbg_flags & 16) == 0) {
                            if (thread_idx == 0) { mbar_init(mbar0, 1); mbar_init(mbar1, 1); cutlass::arch::fence_barrier_init(); }
                            bar_sync(kBarW);
                        }
                        const bool run_chain = (params_ptr->dbg_flags & 1) == 0;
                        if constexpr (MaxRows == 1) {
                        // M = 1, CUDA cores only. One TMA bulk copy of the tagged z (all ranks) into smem; per LUT
                        // group of kG ranks: build the U nibble LUTs [kNq][16 patterns] in the smem_vw region (task =
                        // (LUT, pattern quad): one float4 store), then thread (column c, nibble half h) does one
                        // LDS + one FADD per nibble for its 4 nibbles of every word in the group.
                        constexpr int kG = kStage;
                        constexpr int kNq = kG / 4;
                        constexpr int kRowStride = kNq * 16;
                        constexpr int kWordsG = (kG + 31) / 32;
                        constexpr int kTasks = (kNq * 4 + kThreads - 1) / kThreads;
                        static_assert(kRowStride <= kMaxChunk * 128 * 2, "U LUT exceeds the smem_vw region");
                        const int c = thread_idx & 63;
                        const int h = thread_idx >> 6;
                        float acc = 0.f;
                        float4 const* zt = reinterpret_cast<float4 const*>(smem_z32);          // tagged pairs [kStage]
                        float4* ulut4 = reinterpret_cast<float4*>(smem_vw);
                        const uint32_t lut_base = smem_u32(smem_vw);
                        const uint32_t sel0 = 0x4440u | static_cast<uint32_t>(2 * h), sel1 = 0x4440u | static_cast<uint32_t>(2 * h + 1);
                        // pass 0 (optional, m2_warm): the LUT build + lookups on stale smem before the poll,
                        // so the post-poll code runs with a warm instruction cache
                        if constexpr (Mode == 6) {
                            if (run_chain) {
                                poll_stage_all(rows, 0, kStage, mbar0, zphase);
                                stamp(1);
                                float a0 = 0.f, a1 = 0.f;
                                CUTLASS_PRAGMA_UNROLL
                                for (int i = 0; i < RankStatic / 2; i += 2) {
                                    a0 = fmaf(ud[i], smem_z32[2 * (h * (RankStatic / 2) + i)], a0);
                                    a1 = fmaf(ud[i + 1], smem_z32[2 * (h * (RankStatic / 2) + i + 1)], a1);
                                }
                                acc = a0 + a1;
                            }
                        } else {
                        #pragma unroll 1
                        for (int pass = (run_chain && params_ptr->m2_warm) ? 0 : 1; pass < 2; ++pass) {
                        if (run_chain) {
                            acc = 0.f;
                            if (pass == 1) {
                                poll_stage_all(rows, 0, kStage, mbar0, zphase);
                                stamp(1);
                            } else {
                                bar_sync(kBarW);
                            }
                            CUTLASS_PRAGMA_UNROLL
                            for (int k = 0; k < kTasks; ++k) {
                                const int e = thread_idx + k * kThreads;
                                const int l = e >> 2, quad = e & 3;
                                if (l < kNq) {
                                    const int pi = (4 * l) >> 1;
                                    const float4 ta = zt[pi], tb = zt[pi + 1];
                                    const float z0 = ta.x, z1 = ta.z, z2 = tb.x, z3 = tb.z;
                                    const float hi = ((quad & 1) ? z2 : -z2) + ((quad & 2) ? z3 : -z3);
                                    const float a = z0 + z1, b = z0 - z1;
                                    ulut4[l * 4 + quad] = make_float4(hi - a, hi + b, hi - b, hi + a);   // patterns 4*quad + {0,1,2,3}
                                }
                            }
                            bar_sync(kBarW);
                            CUTLASS_PRAGMA_UNROLL
                            for (int w = 0; w < kWordsG; ++w) {
                                const int qb = w * 8 + 4 * h;
                                if (qb < kNq) {
                                    const uint32_t u = uwr_all[w];
                                    const uint32_t ue = u & 0x0F0F0F0Fu, uo = (u >> 4) & 0x0F0F0F0Fu;
                                    const uint32_t a0 = lut_base + (__byte_perm(ue, 0u, sel0) << 2) + (qb + 0) * 64;
                                    const uint32_t a1 = lut_base + (__byte_perm(uo, 0u, sel0) << 2) + (qb + 1) * 64;
                                    const uint32_t a2 = lut_base + (__byte_perm(ue, 0u, sel1) << 2) + (qb + 2) * 64;
                                    const uint32_t a3 = lut_base + (__byte_perm(uo, 0u, sel1) << 2) + (qb + 3) * 64;
                                    float t0, t1, t2, t3;
                                    asm("ld.shared.f32 %0, [%1];" : "=f"(t0) : "r"(a0));
                                    asm("ld.shared.f32 %0, [%1];" : "=f"(t1) : "r"(a1));
                                    asm("ld.shared.f32 %0, [%1];" : "=f"(t2) : "r"(a2));
                                    asm("ld.shared.f32 %0, [%1];" : "=f"(t3) : "r"(a3));
                                    acc += (t0 + t1) + (t2 + t3);
                                }
                            }
                            if (pass == 0) bar_sync(kBarW);   // warm pass done with the LUT region before the poll overwrites nothing / next build
                        }
                        }
                        }
                        stamp(2);
                        bar_sync(kBarW);
                        stamp(3);
                        const bool okc = (tile_n_start + c) < params_ptr->n_cols;
                        smem_part[h * kTileN + c] = okc ? acc : 0.f;
                        bar_sync(kBarW);
                        if (h == 1) smem_part[c] += smem_part[kTileN + c];
                        stamp(4);
                        bar_sync(kBarW);
                        return;
                        } else {
                        // M > 1, CUDA cores only, row-major: thread = (row m, 8 columns 8*cg..). U for the tile
                        // ([words][64] int32) arrives by TMA into smem_vw; per pass the tagged z is bulk-copied and
                        // compacted to fp32 [rows][kStage]; per 32-rank word: 32 z in registers, 8 u words, 256
                        // predicated adds (sum of +z over set bits; 2*sum - total at the end).
                        constexpr int kWordsP = (kStage + 31) / 32;
                        constexpr int kRw = kStage < 32 ? kStage : 32;                 // ranks per word
                        constexpr int kPasses = RankStatic / kStage;
                        static_assert(RankStatic % kStage == 0, "rank staging must tile evenly");
                        const int m0 = thread_idx >> 3;
                        const int cg = thread_idx & 7;
                        constexpr int kRB = MaxRows > 16 ? MaxRows / 16 : 1;          // row blocks of 16 (thread = row m0 + 16 rb)
                        static_assert(kRB == 1 || kPasses == 1, "32-row mode 2 needs a single rank pass (R <= 128)");
                        float accp_all[kRB][8];
                        float zsum_all[kRB];
                        CUTLASS_PRAGMA_UNROLL
                        for (int rb = 0; rb < kRB; ++rb) {
                            zsum_all[rb] = 0.f;
                            CUTLASS_PRAGMA_UNROLL
                            for (int jj = 0; jj < 8; ++jj) accp_all[rb][jj] = 0.f;
                        }
                        if (run_chain) {
                            if (thread_idx == 0) {
                                mbar_expect_tx(mbar1, static_cast<uint32_t>(kRankWords * 256));
                                CUTLASS_PRAGMA_UNROLL
                                for (int w = 0; w < kRankWords; ++w) {
                                    bulk_g2s(smem_u32(smem_vw) + static_cast<uint32_t>(w * 256), params_ptr->u_rw_ptr + static_cast<size_t>(w) * params_ptr->n_cols + tile_n_start, 256u, mbar1);
                                }
                            }
                            for (int pass = 0; pass < kPasses; ++pass) {
                                poll_stage_all(rows, pass * kStage, kStage, mbar0, zphase);
                                if (pass == 0) { stamp(1); mbar_wait(mbar1, 0u); }
                                CUTLASS_PRAGMA_UNROLL
                                for (int rb = 0; rb < kRB; ++rb) {
                                const int m = m0 + 16 * rb;
                                float (&accp)[8] = accp_all[rb];
                                float& zsum = zsum_all[rb];
                                if (m < rows) {
                                    float const* zrow = smem_z32 + m * kStagePad;
                                    // double-buffered: word w+1's z (8 float4) and u (2 uint4) are loaded while word w is expanded
                                    float4 zb[2][kRw / 4];
                                    uint4 ub[2][2];
                                    {
                                        float4 const* zp0 = reinterpret_cast<float4 const*>(zrow);
                                        CUTLASS_PRAGMA_UNROLL
                                        for (int q = 0; q < kRw / 4; ++q) zb[0][q] = zp0[q];
                                        uint4 const* up0 = reinterpret_cast<uint4 const*>(smem_vw + (pass * kWordsP) * 64 + 8 * cg);
                                        ub[0][0] = up0[0]; ub[0][1] = up0[1];
                                    }
                                    const int dbg = params_ptr->dbg_flags;
                                    auto expand_word = [&](float4 const (&zc)[kRw / 4], uint4 const (&uc)[2]) {
                                        const uint4 ua = uc[0], ubv = uc[1];
                                        const uint32_t uw[8] = {ua.x, ua.y, ua.z, ua.w, ubv.x, ubv.y, ubv.z, ubv.w};
                                        float z[kRw];
                                        CUTLASS_PRAGMA_UNROLL
                                        for (int q = 0; q < kRw / 4; ++q) {
                                            const float4 zv = zc[q];
                                            z[4 * q] = zv.x; z[4 * q + 1] = zv.y; z[4 * q + 2] = zv.z; z[4 * q + 3] = zv.w;
                                        }
                                        if (dbg & 32) {                                   // loads only
                                            CUTLASS_PRAGMA_UNROLL
                                            for (int jj = 0; jj < 8; ++jj) accp[jj] += z[jj] + __uint_as_float(uw[jj] & 1u);
                                            return;
                                        }
                                        // predicated adds (sum of +z over set bits; corr = 2*sum - total): ~3.7 cycles/element,
                                        // the fastest exact CUDA-core form measured on B300 (FSEL 3.6, PRMT+FFMA 5.2, FFMA2 5.2)
                                        CUTLASS_PRAGMA_UNROLL
                                        for (int r = 0; r < kRw; ++r) zsum += z[r];
                                        CUTLASS_PRAGMA_UNROLL
                                        for (int jj = 0; jj < 8; ++jj) {
                                            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                                            CUTLASS_PRAGMA_UNROLL
                                            for (int r = 0; r < kRw; r += 4) {
                                                pred_add(d0, uw[jj] & (1u << r), z[r]);
                                                pred_add(d1, uw[jj] & (1u << (r + 1)), z[r + 1]);
                                                pred_add(d2, uw[jj] & (1u << (r + 2)), z[r + 2]);
                                                pred_add(d3, uw[jj] & (1u << (r + 3)), z[r + 3]);
                                            }
                                            accp[jj] += (d0 + d1) + (d2 + d3);
                                        }
                                    };
                                    auto prefetch_word = [&](int wn, float4 (&zn)[kRw / 4], uint4 (&un)[2]) {
                                        float4 const* zpn = reinterpret_cast<float4 const*>(zrow + 32 * wn);
                                        if (dbg & 64) {                                   // no z loads (arithmetic only)
                                            CUTLASS_PRAGMA_UNROLL
                                            for (int q = 0; q < kRw / 4; ++q) zn[q] = make_float4(__int_as_float(wn + q), 1.f, 2.f, 3.f);
                                        } else {
                                            CUTLASS_PRAGMA_UNROLL
                                            for (int q = 0; q < kRw / 4; ++q) zn[q] = zpn[q];
                                        }
                                        uint4 const* upn = reinterpret_cast<uint4 const*>(smem_vw + (pass * kWordsP + wn) * 64 + 8 * cg);
                                        un[0] = upn[0]; un[1] = upn[1];
                                    };
                                    for (int w = 0; w < kWordsP; w += 2) {
                                        if (w + 1 < kWordsP) prefetch_word(w + 1, zb[1], ub[1]);
                                        expand_word(zb[0], ub[0]);
                                        if (w + 1 < kWordsP) {
                                            if (w + 2 < kWordsP) prefetch_word(w + 2, zb[0], ub[0]);
                                            expand_word(zb[1], ub[1]);
                                        }
                                    }
                                }
                                }
                            }
                        }
                        stamp(2);
                        bar_sync(kBarW);
                        stamp(3);
                        CUTLASS_PRAGMA_UNROLL
                        for (int rb = 0; rb < kRB; ++rb) {
                        const int m = m0 + 16 * rb;
                        float (&accp)[8] = accp_all[rb];
                        const float zsum = zsum_all[rb];
                        if (m < rows) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int jj = 0; jj < 8; ++jj) {
                                const int col = 8 * cg + jj;
                                const bool okc = (tile_n_start + col) < params_ptr->n_cols;
                                smem_part[m * kTileN + col] = okc ? fmaf(2.f, accp[jj], -zsum) : 0.f;
                            }
                        }
                        }
                        stamp(4);
                        bar_sync(kBarW);
                        return;
                        }
                    }
                } else if constexpr (Mode == 4) {
                    stamp(0);
                    // pass 0 (optional): the expansion on stale smem, so its instructions are cached when z lands
                    #pragma unroll 1
                    for (int pass = params_ptr->tile_warm ? 0 : 1; pass < 2; ++pass) {
                        if constexpr (MaxRows == 2) {
                            const bool cheap = (pass == 0) && params_ptr->warm_cheap;
                            am_ = cheap ? 0 : -1;
                            wr_ = !cheap;
                        }
                        if (pass == 1) {
                            if constexpr (MaxRows == 16) {
                                if (params_ptr->side_corr) {
                                    if (!(params_ptr->dbg_flags & 1)) poll_corr_rows(rows);
                                    stamp(3);
                                    bar_sync(kBarW);   // smem_part (final correction) -> visits
                                    break;
                                }
                            }
                            if (!(params_ptr->dbg_flags & 1)) {                    // dbg 1: no wait (timing only)
                                if (!kLeanTile && params_ptr->poll_delay_ns > 0) {
                                    // start polling late: fewer poll loads compete with the weight stream in L2
                                    const int64_t t_end = globaltimer() + params_ptr->poll_delay_ns;
                                    while (globaltimer() < t_end) __nanosleep(128);
                                }
                                if constexpr (kLeanTile) poll_z_tagged(rows);
                                else {
                                    if (params_ptr->z4b) poll_z_tagged4(rows);
                                    else poll_z_tagged(rows);
                                }
                            }
                            if constexpr (MaxRows == 2) {
                                if (kV2Only || params_ptr->warm_cheap) {
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int w = 0; w < kRankWords; ++w) uwa[w] = upend_[w];
                                }
                            } else if constexpr (MaxRows == 8 && RankStatic <= 128) {   // (r > 128: +16 live registers, M8 r512 spills)
                                CUTLASS_PRAGMA_UNROLL
                                for (int w = 0; w < kRankWords; ++w) uwa[w] = upend_[w];
                            }
                            stamp(3);
                        }
                        bar_sync(kBarW);
                        if constexpr (MaxRows == 1) {
                            if (params_ptr->tile_lut) expand_lut_row0(uwr);
                            else expand_fsel_row0(uwr);
                        } else if constexpr (MaxRows == 2) {
                            if constexpr (kV2Only) expand_lut_2rows_v2(rows, uwa);
                            else if constexpr (kV1Only) expand_lut_2rows(rows, uwa, pass == 1 && Arguments::help_path(*params_ptr) == 1);
                            else {
                                if (params_ptr->exp2v2) expand_lut_2rows_v2(rows, uwa);
                                else expand_lut_2rows(rows, uwa, pass == 1 && Arguments::help_path(*params_ptr) == 1);
                            }
                        } else if constexpr (MaxRows == 4 || MaxRows == 8) {
                            if constexpr (MaxRows == 4) {
                                if (rows <= 2 && params_ptr->exp2r) expand_lut_2rows(rows, uwa, pass == 1 && Arguments::help_path(*params_ptr) == 1);
                                else if constexpr (RankStatic >= 512) {
                                    if (params_ptr->exp4r16) expand_lut_4rows16(rows, uwa, pass == 1 && Arguments::help_path(*params_ptr) == 2);
                                    else expand_lut_rowsK<MaxRows>(rows, uwa);
                                } else {
                                    expand_lut_rowsK<MaxRows>(rows, uwa);
                                }
                            } else {
                                expand_lut_rowsK<MaxRows>(rows, uwa);
                            }
                        } else {
                            if (params_ptr->exp2) expand_lut_rows2(rows, uwa);
                            else expand_lut_rows(rows, uwr);
                        }
                        bar_sync(kBarW);
                    }
                    stamp(4);
                } else {
                    if (params_ptr->pdl_wait) {
                        asm volatile("griddepcontrol.wait;" ::: "memory");   // z kernel (PDL primary) complete + flushed
                    }
                    stage_external_z(rows);
                }
                if constexpr (!kInKernelZ && Mode != 4) {
                bar_sync(kBarW);
                if constexpr (MaxRows == 1) {
                    expand_fsel_row0(uwr);
                } else {
                    expand_mma(rows, uwa, uwb);
                }
                stamp(4);
                bar_sync(kBarW);   // expansion writes -> visits (explicit; the r16 race showed begin_sync alone is not enough)
                }
                // the collective's begin_sync (named barrier over 128 threads) publishes smem_part
            }

            CUTLASS_DEVICE void
            end() {
                stamp(6);
            }

            CUTLASS_DEVICE void
            begin_loop(int epi_m, int epi_n) {
                auto coord_vec = coalesce(coords(_, _, _, epi_m, epi_n));
                auto coord0 = coord_vec(0);
                my_row = int(get<0>(coord0)) - tile_m_start;
                sub_n0 = int(get<1>(coord0)) - tile_n_start;
                my_row_live = (my_row >= 0) && (my_row < rows_valid);
            }

            template <typename ElementAccumulator, int FragmentSize>
            CUTLASS_DEVICE cutlass::Array<ElementCompute, FragmentSize>
            visit(cutlass::Array<ElementAccumulator, FragmentSize> const&,
                  int epi_v, int epi_m, int epi_n) {
                cutlass::Array<ElementCompute, FragmentSize> out;
                if constexpr (Mode == 2) {
                    if (!stamped_visit) {
                        stamped_visit = true;
                        stamp(5);
                    }
                }
                if (!my_row_live) {
                    out.fill(ElementCompute(0));
                    return out;
                }
                // fragments of a subtile are contiguous in n for this thread (32x32b T2R layout):
                // n = sub_n0 + epi_v * FragmentSize; columns past N are never visited for N % 64 == 0
                const int cn = sub_n0 + epi_v * FragmentSize;
                float const* src = smem_part + my_row * kTileN + cn;
                if constexpr (FragmentSize % 4 == 0) {
                    if (cn + FragmentSize <= kTileN) {
                        float4 const* s4 = reinterpret_cast<float4 const*>(src);
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < FragmentSize / 4; ++i) {
                            const float4 v = s4[i];
                            out[4 * i] = v.x; out[4 * i + 1] = v.y; out[4 * i + 2] = v.z; out[4 * i + 3] = v.w;
                        }
                        return out;
                    }
                }
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < FragmentSize; ++i) {
                    out[i] = (cn + i < kTileN) ? src[i] : ElementCompute(0);
                }
                return out;
            }
        };

        template <bool ReferenceSrc, class... Args>
        CUTLASS_DEVICE auto
        get_consumer_store_callbacks(
            cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args) {
            auto [M, N, K, L] = args.problem_shape_mnkl;
            auto [m, n, k, l] = args.tile_coord_mnkl;
            (void)k;
            (void)l;
            auto problem_shape_mnl = make_shape(M, N, L);
            auto coord_tensor = make_identity_tensor(problem_shape_mnl);
            auto tC_coord = cutlass::epilogue::fusion::sm90_partition_for_epilogue<ReferenceSrc>(
                coord_tensor, args.tile_shape_mnk, args.tile_coord_mnkl,
                args.epi_tile, args.tiled_copy, args.thread_idx);
            const int tile_m_start = int(m) * int(size<0>(args.tile_shape_mnk));
            const int tile_n_start = int(n) * kTileN;
            return ConsumerStoreCallbacks<decltype(tC_coord), decltype(problem_shape_mnl)>(
                cute::move(tC_coord), problem_shape_mnl, params_ptr, smem_part, smem_z16, smem_z32, smem_lut4, smem_lut, smem_red, smem_vw,
                tile_m_start, tile_n_start, int(n), args.thread_idx);
        }
    };

    // Side-CTA entry (extra grid layer). One side CTA = one work item: rank chunk
    // [8*side_idx, +8) x all rows (<= 16) x full K, on the SM's idle tensor cores:
    // the x rows are staged in smem chunk by chunk (cooperative coalesced loads),
    // each of the 8 warps reduces a K/8 range with bf16 m16n8k16 mma (fp32 acc,
    // K permuted so each lane's A fragment is one 8-byte smem load per row), the
    // per-warp C fragments are scaled by S and atomically added into the parity
    // buffer z[epoch&1], one release-add on the arrival counter per item, then the
    // item zeroes its entries of the other parity buffer for the next launch.
    template <class ElementCompute, int RankStatic, int TileN, int MaxRows, int Mode>
    CUTLASS_DEVICE void
    Sm90BinaryZUCwFetch<ElementCompute, RankStatic, TileN, MaxRows, Mode>::Arguments::side_cta(
        Arguments const& p, char* smem, int side_idx) {
        if constexpr (Mode == 5) {
            if (side_idx >= p.num_side || threadIdx.x >= 256) return;
            bsvd_br::run(p.br, smem, side_idx, p.num_side + (p.br_tiles_work ? p.br_tiles : 0), nullptr);
            return;
        }
        // lean side code for the 2- and 8-row kernels: only the rank-inner worker (zw_mode 6) is compiled in (the side
        // SMs fetch the side program cold every launch; unused workers only dilute its layout)
        constexpr bool kLeanSide = (MaxRows == 2 || MaxRows == 8);
        if constexpr (Mode == 4 && (MaxRows == 2 || MaxRows == 4 || MaxRows == 8)) {
            if constexpr (!kLeanSide) {
            if (p.zw_mode == 4 && MaxRows != 2) {
                // one side CTA = one rank block of zw_rc ranks for all rows
                if (side_idx >= p.num_side || side_idx >= RankStatic / p.zw_rc || threadIdx.x >= 256) return;
                int64_t* st = (BSVD_STAMPS && p.diag_ptr != nullptr && side_idx < 64) ? p.diag_ptr + 2048 + side_idx * 16 : nullptr;
                if (st != nullptr && threadIdx.x == 0) st[0] = globaltimer();
                bsvd_br::zworker_rows<(MaxRows == 2 ? 4 : MaxRows)>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, p.zw_rc, side_idx,
                                               reinterpret_cast<float2*>(p.z_ptr), p.epoch, smem, static_cast<int>(threadIdx.x),
                                               p.z_copies, p.m_rows * RankStatic, st);
                return;
            }
            if (p.zw_mode == 5) {
                // one side CTA = one rank block of zw_rc ranks for all rows, direct from bf16 (no LUT)
                if (side_idx >= p.num_side || side_idx >= RankStatic / p.zw_rc || threadIdx.x >= 256) return;
                int64_t* st = (BSVD_STAMPS && p.diag_ptr != nullptr && side_idx < 64) ? p.diag_ptr + 2048 + side_idx * 16 : nullptr;
                if (st != nullptr && threadIdx.x == 0) st[0] = globaltimer();
                float2* zt = reinterpret_cast<float2*>(p.z_ptr);
                if (p.m_rows <= 2) {
                    bsvd_br::zworker_direct_rows<2>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, p.zw_rc, side_idx, zt, p.epoch,
                                                    smem, static_cast<int>(threadIdx.x), p.z_copies, p.m_rows * RankStatic, st);
                } else {
                    bsvd_br::zworker_direct_rows<MaxRows>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, p.zw_rc, side_idx, zt,
                                                          p.epoch, smem, static_cast<int>(threadIdx.x), p.z_copies, p.m_rows * RankStatic, st);
                }
                return;
            }
            }   // !kLeanSide
            if (p.zw_mode == 6) {
                // one side CTA = RankStatic / 64 ranks for all rows, rank-inner (x converted once per nibble)
                constexpr int kRc6 = RankStatic / 64 < 1 ? 1 : (RankStatic / 64 > 8 ? 8 : RankStatic / 64);
                if (side_idx >= p.num_side || side_idx >= RankStatic / kRc6 || threadIdx.x >= 256) return;
                int64_t* st = (BSVD_STAMPS && p.diag_ptr != nullptr && side_idx < 64) ? p.diag_ptr + 2048 + side_idx * 16 : nullptr;
                if (st != nullptr && threadIdx.x == 0) {
                    st[0] = globaltimer();
                    uint32_t smid;
                    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
                    st[15] = smid;
                }
                float2* zt = reinterpret_cast<float2*>(p.z_ptr);
                const int rb6 = (side_idx + p.side_rot) % (RankStatic / kRc6);   // side_rot: diagnostic rotation of rank blocks
                if constexpr (MaxRows == 2) {
                    // lean 2-row kernel: only the default worker is compiled (dead variants cost instruction fetch)
                    bsvd_br::zworker_ranks_inner<2, kRc6>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, rb6, zt, p.epoch,
                                                          smem, static_cast<int>(threadIdx.x), p.z_copies, p.m_rows * RankStatic, st);
                } else if constexpr (MaxRows == 8) {
                    // lean 8-row kernel: only the packed-FMA rank-inner worker (m_rows 5-8)
                    bsvd_br::zworker_ranks_inner<8, kRc6>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, rb6, zt, p.epoch,
                                                          smem, static_cast<int>(threadIdx.x), p.z_copies, p.m_rows * RankStatic, st, true);
                } else if (p.m_rows <= 2) {
                    // side_unroll: nibble-loop unroll of the scalar-FMA path (instruction-fetch vs load-latency trade)
                    if (p.side_unroll == 1) {
                        bsvd_br::zworker_ranks_inner<2, kRc6, 1>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, rb6, zt, p.epoch,
                                                                 smem, static_cast<int>(threadIdx.x), p.z_copies, p.m_rows * RankStatic, st);
                    } else if (p.side_unroll == 2) {
                        bsvd_br::zworker_ranks_inner<2, kRc6, 2>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, rb6, zt, p.epoch,
                                                                 smem, static_cast<int>(threadIdx.x), p.z_copies, p.m_rows * RankStatic, st);
                    } else {
                        bsvd_br::zworker_ranks_inner<2, kRc6>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, rb6, zt, p.epoch,
                                                              smem, static_cast<int>(threadIdx.x), p.z_copies, p.m_rows * RankStatic, st);
                    }
                } else {
                    bsvd_br::zworker_ranks_inner<MaxRows, kRc6>(p.x_ptr, p.v_ptr, p.s_ptr, p.m_rows, p.k_cols, RankStatic, rb6, zt, p.epoch,
                                                                smem, static_cast<int>(threadIdx.x), p.z_copies, p.m_rows * RankStatic, st, p.side_f2 != 0);
                }
                return;
            }
        }
        if constexpr (Mode == 4 && !kLeanSide) {
            const int per_row = RankStatic / p.zw_rc;
            if (side_idx >= p.num_side || side_idx >= p.m_rows * per_row || threadIdx.x >= 256 || (p.dbg_flags & 2)) return;
            const int m = side_idx / per_row, rb = side_idx - m * per_row;
            int64_t* st = (BSVD_STAMPS && p.diag_ptr != nullptr && side_idx < 64) ? p.diag_ptr + 2048 + side_idx * 16 : nullptr;
            if (st != nullptr && threadIdx.x == 0) st[0] = globaltimer();
            bsvd_zworker<RankStatic>(p.x_ptr + static_cast<size_t>(m) * p.k_cols, p.v_ptr, p.s_ptr, p.k_cols, p.zw_rc, rb,
                                     reinterpret_cast<float2*>(p.z_ptr) + static_cast<size_t>(m) * RankStatic, p.epoch, smem,
                                     static_cast<int>(threadIdx.x), st, p.z_copies, p.m_rows * RankStatic, p.zw_mode,
                                     p.u_col32_ptr, p.n_cols, p.side_corr ? reinterpret_cast<uint32_t*>(p.z_ptr) : nullptr, m,
                                     p.z4b ? reinterpret_cast<uint32_t*>(p.z_ptr) + static_cast<size_t>(m) * RankStatic : nullptr);
            return;
        }
        if constexpr (Mode == 6) {
            using Fetch = Sm90BinaryZUCwFetch<ElementCompute, RankStatic, TileN, MaxRows, Mode>;
            using CB = typename Fetch::template ConsumerStoreCallbacks<int, int>;
            if (side_idx >= p.num_side || (p.dbg_flags & 2)) return;
            const int n_units = p.m_rows * RankStatic;               // one rank per unit
            for (int u = side_idx; u < n_units; u += p.num_side) CB::z_unit_dense(p, u / RankStatic, u % RankStatic, static_cast<int>(threadIdx.x), smem);
            return;
        }
        if constexpr (Mode == 2) {
            using Fetch = Sm90BinaryZUCwFetch<ElementCompute, RankStatic, TileN, MaxRows, Mode>;
            using CB = typename Fetch::template ConsumerStoreCallbacks<int, int>;
            if (p.diag_ptr != nullptr && threadIdx.x == 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(p.diag_ptr + 4087), 1ull);
            }
            if (p.side_kind == 0) {
                // v3 units (row, rc ranks): one unit at a time per side CTA (both 128-thread groups), round-robin
                const int rc = p.side_rc;
                if (side_idx >= p.num_side || (p.dbg_flags & 2)) return;
                const int n_rc = RankStatic / rc;
                const int n_units = p.m_rows * n_rc;
                for (int u = side_idx; u < n_units; u += p.num_side) {
                    const int m = u / n_rc;
                    const int i0 = (u - m * n_rc) * rc;
                    int64_t* stu = (p.diag_ptr != nullptr && u < 136) ? p.diag_ptr + 2048 + u * 8 : nullptr;
                    if (stu != nullptr && threadIdx.x == 0) stu[0] = Fetch::globaltimer();
                    CB::z_unit3(p, m, i0, rc, static_cast<int>(threadIdx.x), smem, stu);
                }
                return;
            }
            // (tensor-core side path removed: the binary branch runs on CUDA cores only)
        }
    }

    // Adapted from example 72b
    template <typename ElementA,
              typename MmaTileShape = Shape<_128, _128, _256>,
              typename ClusterShape = Shape<_2, _4, _1>,
              typename KernelMainloopPolicy = cutlass::gemm::collective::KernelScheduleAuto,
              int AlignmentA = 32,
              typename ElementB = ElementA,
              int AlignmentB = 32,
              typename LayoutATag = cutlass::layout::RowMajor,
              typename LayoutBTag = cutlass::layout::ColumnMajor,
              typename ElementD = cutlass::bfloat16_t,
              typename ArchTag = cutlass::arch::Sm100>
    torch::Tensor gemm_fp4fp4_accum_fp32(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor const &A_sf, torch::Tensor const &B_sf, torch::Tensor const &alpha)
    {
        // C/D matrix configuration
        using ElementC = void;                        // Element type for C matrix operand
        using LayoutCTag = cutlass::layout::RowMajor; // Layout type for C matrix operand
        using LayoutDTag = cutlass::layout::RowMajor; // Layout type for D matrix operand

        constexpr int AlignmentC = 1;                                           // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
        constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value; // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)

        // Kernel functional config
        using ElementAccumulator = float;                                // Element type for internal accumulation
        using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp; // Operator class tag

        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            MmaTileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator, ElementAccumulator,
            ElementC, LayoutCTag, AlignmentC,
            ElementD, LayoutDTag, AlignmentD,
            cutlass::epilogue::collective::EpilogueScheduleAuto // Epilogue schedule policy
            >::CollectiveOp;

        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            ElementA, LayoutATag, AlignmentA,
            ElementB, LayoutBTag, AlignmentB,
            ElementAccumulator,
            MmaTileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelMainloopPolicy>::CollectiveOp;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
            Shape<int, int, int, int>, // Indicates ProblemShape
            CollectiveMainloop,
            CollectiveEpilogue,
            void>;

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
        {
            static bool printed_floor = false;
            if (!printed_floor && getenv("BILOCO_VERBOSE") != nullptr) {
                printed_floor = true;
                printf("[accum_fp32 floor tileN=%d] smem=%d stages=%d\n", static_cast<int>(cute::size<1>(typename Gemm::GemmKernel::TileShape{})), static_cast<int>(Gemm::GemmKernel::SharedStorageSize), static_cast<int>(Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages));
            }
        }

        // Reference device GEMM implementation type
        using StrideA = typename Gemm::GemmKernel::StrideA;
        using StrideB = typename Gemm::GemmKernel::StrideB;
        using StrideC = typename Gemm::GemmKernel::StrideC;
        using StrideD = typename Gemm::GemmKernel::StrideD;

        using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA; // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
        using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB; // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.

        using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

        // torch::checkAllContiguous("gemm_fp4fp4_accum_fp32_out_bf16", {{A, "A", 0}, {B, "B", 1}, {A_sf, "A_sf", 2}, {B_sf, "B_sf", 3}, {alpha, "alpha", 4}});
        torch::checkDeviceType("gemm_fp4fp4_accum_fp32_out_bf16", {A, B, A_sf, B_sf, alpha}, at::DeviceType::CUDA);
        torch::checkAllSameGPU("gemm_fp4fp4_accum_fp32_out_bf16", {{A, "A", 0}, {B, "B", 1}, {A_sf, "A_sf", 2}, {B_sf, "B_sf", 3}, {alpha, "alpha", 4}});

        check_block_scale_factor_type<ElementA>(A_sf, "A_sf");
        check_block_scale_factor_type<ElementB>(B_sf, "B_sf");

        auto [M, N, K, L] = check_and_get_fp4_matmul_dims<ElementA, LayoutATag, ElementB, LayoutBTag>(A, B, A_sf, B_sf);
        auto D = torch::empty({L, M, N}, torch::dtype(element_traits<ElementD>::scalar_type).device(A.device()));

        Gemm gemm;

        // Create stride and layout information for the packed tensors
        // For packed NVFP4 tensors, we need to use the appropriate stride and layout
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, L});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});

        // Create scale factor layouts
        LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, L));
        LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, L));

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, L},
            {// Mainloop arguments
             static_cast<typename ElementA::DataType const *>(A.data_ptr()), stride_A,
             static_cast<typename ElementB::DataType const *>(B.data_ptr()), stride_B,
             static_cast<typename ElementA::ScaleFactorType const *>(A_sf.data_ptr()), layout_SFA,
             static_cast<typename ElementB::ScaleFactorType const *>(B_sf.data_ptr()), layout_SFB},
            {// Epilogue arguments
             {1.0f, 0.0f},
             nullptr,
             stride_C,
             static_cast<ElementD *>(D.data_ptr()),
             stride_D}};

        args.epilogue.thread.alpha_ptr = static_cast<ElementAccumulator *>(alpha.data_ptr());

        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(args));

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        auto workspace_size = Gemm::get_workspace_size(args);
        auto workspace = torch::empty(
            {static_cast<int64_t>(workspace_size)},
            torch::dtype(torch::kUInt8).device(A.device()));
        void* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;

        // Newer CUTLASS schedulers can require non-empty workspace during
        // initialization. Passing it explicitly keeps native FP4 working
        // across both B200 (sm_100a) and B300 (sm_103a) builds.
        CUTLASS_CHECK(gemm.initialize(args, workspace_ptr, stream));

        CUTLASS_CHECK(gemm.run(stream));

        return D;
    }

    // Variant of gemm_fp4fp4_accum_fp32 that accepts a residual tensor C (same
    // shape as the output D). Computes D = alpha * (A @ B) + 1.0 * C.
    //
    // Used by the binary low-rank recipe to fuse the final
    //   y = fp4_main + y_branch
    // addition into the FP4 GEMM's epilogue, eliminating one kernel launch
    // from the per-layer chain.
    template <typename ElementA,
              typename MmaTileShape = Shape<_128, _128, _256>,
              typename ClusterShape = Shape<_2, _4, _1>,
              typename KernelMainloopPolicy = cutlass::gemm::collective::KernelScheduleAuto,
              int AlignmentA = 32,
              typename ElementB = ElementA,
              int AlignmentB = 32,
              typename LayoutATag = cutlass::layout::RowMajor,
              typename LayoutBTag = cutlass::layout::ColumnMajor,
              typename ElementD = cutlass::bfloat16_t,
              typename ElementC = ElementD,
              typename ArchTag = cutlass::arch::Sm100>
    torch::Tensor gemm_fp4fp4_residual_accum_fp32(torch::Tensor const &A,
                                                   torch::Tensor const &B,
                                                   torch::Tensor const &A_sf,
                                                   torch::Tensor const &B_sf,
                                                   torch::Tensor const &alpha,
                                                   torch::Tensor const &C_residual)
    {
        using LayoutCTag = cutlass::layout::RowMajor;
        using LayoutDTag = cutlass::layout::RowMajor;

        constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
        constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

        using ElementAccumulator = float;
        using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            MmaTileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator, ElementAccumulator,
            ElementC, LayoutCTag, AlignmentC,
            ElementD, LayoutDTag, AlignmentD,
            cutlass::epilogue::collective::EpilogueScheduleAuto
            >::CollectiveOp;

        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            ElementA, LayoutATag, AlignmentA,
            ElementB, LayoutBTag, AlignmentB,
            ElementAccumulator,
            MmaTileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelMainloopPolicy>::CollectiveOp;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
            Shape<int, int, int, int>,
            CollectiveMainloop,
            CollectiveEpilogue,
            void>;

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

        using StrideA = typename Gemm::GemmKernel::StrideA;
        using StrideB = typename Gemm::GemmKernel::StrideB;
        using StrideC = typename Gemm::GemmKernel::StrideC;
        using StrideD = typename Gemm::GemmKernel::StrideD;
        using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
        using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
        using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

        torch::checkDeviceType("gemm_fp4fp4_residual",
            {A, B, A_sf, B_sf, alpha, C_residual}, at::DeviceType::CUDA);

        check_block_scale_factor_type<ElementA>(A_sf, "A_sf");
        check_block_scale_factor_type<ElementB>(B_sf, "B_sf");

        auto [M, N, K, L] = check_and_get_fp4_matmul_dims<ElementA, LayoutATag, ElementB, LayoutBTag>(A, B, A_sf, B_sf);
        TORCH_CHECK(C_residual.scalar_type() == element_traits<ElementC>::scalar_type,
                    "C_residual must match output dtype (bf16)");
        TORCH_CHECK(C_residual.dim() == 2 && C_residual.size(0) == M && C_residual.size(1) == N,
                    "C_residual must be (M, N) bf16; got shape mismatch");

        auto D = torch::empty({L, M, N}, torch::dtype(element_traits<ElementD>::scalar_type).device(A.device()));

        Gemm gemm;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, L});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});

        LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, L));
        LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, L));

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, L},
            {static_cast<typename ElementA::DataType const *>(A.data_ptr()), stride_A,
             static_cast<typename ElementB::DataType const *>(B.data_ptr()), stride_B,
             static_cast<typename ElementA::ScaleFactorType const *>(A_sf.data_ptr()), layout_SFA,
             static_cast<typename ElementB::ScaleFactorType const *>(B_sf.data_ptr()), layout_SFB},
            {// Epilogue: alpha (set via alpha_ptr below) and beta=1.0 for residual.
             {1.0f, 1.0f},
             static_cast<ElementC *>(C_residual.data_ptr()),
             stride_C,
             static_cast<ElementD *>(D.data_ptr()),
             stride_D}};

        args.epilogue.thread.alpha_ptr = static_cast<ElementAccumulator *>(alpha.data_ptr());

        CUTLASS_CHECK(gemm.can_implement(args));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        auto workspace_size = Gemm::get_workspace_size(args);
        auto workspace = torch::empty(
            {static_cast<int64_t>(workspace_size)},
            torch::dtype(torch::kUInt8).device(A.device()));
        void* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;
        CUTLASS_CHECK(gemm.initialize(args, workspace_ptr, stream));
        CUTLASS_CHECK(gemm.run(stream));

        return D;
    }

    // Stock smalln NVFP4 GEMM + consumer-warp zU (external z or in-kernel z).
    template <int StaticRank, int MaxRows, int Mode, int StagesOverride = 0>
    torch::Tensor gemm_fp4fp4_binary_zu_cw_smalln_impl(torch::Tensor const &A,
                                                       torch::Tensor const &B,
                                                       torch::Tensor const &A_sf,
                                                       torch::Tensor const &B_sf,
                                                       torch::Tensor const &alpha,
                                                       torch::Tensor const &z,
                                                       torch::Tensor const &U_col32,
                                                       int64_t m_rows,
                                                       torch::Tensor const &X_bf16,
                                                       torch::Tensor const &V_packed,
                                                       torch::Tensor const &S,
                                                       torch::Tensor const &Flags,
                                                       int64_t epoch,
                                                       int64_t spin_iters = 0,
                                                       int64_t chase_iters = 0,
                                                       torch::Tensor const &Diag = torch::Tensor(),
                                                       int64_t diag_kind = 0,
                                                       bool pdl = false,
                                                       torch::Tensor const &V_perm = torch::Tensor(),
                                                       torch::Tensor const &U_rw = torch::Tensor(),
                                                       torch::Tensor const &Corr = torch::Tensor())
    {
        using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
        using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
        using ElementC = void;
        using ElementD = cutlass::bfloat16_t;
        using LayoutATag = cutlass::layout::RowMajor;
        using LayoutBTag = cutlass::layout::ColumnMajor;
        using LayoutCTag = cutlass::layout::RowMajor;
        using LayoutDTag = cutlass::layout::RowMajor;
        using MmaTileShape = Shape<_128, _64, _256>;
        using ClusterShape = Shape<_1, _1, _1>;
        using ArchTag = cutlass::arch::Sm100;
        using KernelMainloopPolicy = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;

        constexpr int AlignmentA = 32;
        constexpr int AlignmentB = 32;
        constexpr int AlignmentC = 1;
        constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

        using ElementAccumulator = float;
        using ElementCompute = float;
        using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
        using ElementScalar = float;
        static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

        using AccScaled = cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
            cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>,
            cutlass::epilogue::fusion::Sm90AccFetch>;

        constexpr bool InKernelZ = (Mode == 2 || Mode == 6);
        using BinaryFetch =
            Sm90BinaryZUCwFetch<ElementCompute, StaticRank, 64, MaxRows, Mode>;

        using FusionCallbacks = cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::plus, ElementD, ElementCompute, RoundStyle>,
            AccScaled,
            BinaryFetch>;

        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            MmaTileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator, ElementCompute,
            ElementC, LayoutCTag, AlignmentC,
            ElementD, LayoutDTag, AlignmentD,
            cutlass::epilogue::collective::EpilogueScheduleAuto,
            FusionCallbacks
            >::CollectiveOp;

        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            ElementA, LayoutATag, AlignmentA,
            ElementB, LayoutBTag, AlignmentB,
            ElementAccumulator,
            MmaTileShape, ClusterShape,
            cute::conditional_t<StagesOverride == 0, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, cutlass::gemm::collective::StageCount<StagesOverride>>,
            KernelMainloopPolicy>::CollectiveOp;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
            Shape<int, int, int, int>,
            CollectiveMainloop,
            CollectiveEpilogue,
            void>;

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
        using StrideA = typename Gemm::GemmKernel::StrideA;
        using StrideB = typename Gemm::GemmKernel::StrideB;
        using StrideC = typename Gemm::GemmKernel::StrideC;
        using StrideD = typename Gemm::GemmKernel::StrideD;
        using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
        using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
        using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

        torch::checkDeviceType("gemm_fp4fp4_binary_zu_cw",
            {A, B, A_sf, B_sf, alpha, z, U_col32}, at::DeviceType::CUDA);
        check_block_scale_factor_type<ElementA>(A_sf, "A_sf");
        check_block_scale_factor_type<ElementB>(B_sf, "B_sf");

        auto [M, N, K, L] = check_and_get_fp4_matmul_dims<ElementA, LayoutATag, ElementB, LayoutBTag>(A, B, A_sf, B_sf);
        TORCH_CHECK(L == 1, "zu_cw: batch dim must be 1");
        TORCH_CHECK(M <= 128, "zu_cw: one M tile only (M <= 128)");
        TORCH_CHECK(alpha.scalar_type() == at::kFloat && alpha.numel() == 1, "alpha must be a single fp32");
        if constexpr (InKernelZ) {
            TORCH_CHECK(z.scalar_type() == at::kFloat && z.dim() == 4 && z.size(0) == 8 && z.size(3) == 2 && z.is_contiguous(), "Z_scratch must be fp32 [8, m_rows, R, 2] contiguous");
        } else {
            TORCH_CHECK(z.scalar_type() == at::kFloat && z.dim() == 2 && z.is_contiguous(), "z must be fp32 2D contiguous");
        }
        TORCH_CHECK(U_col32.scalar_type() == at::kInt && U_col32.dim() == 2 && U_col32.is_contiguous(), "U_col32 must be int32 2D contiguous");
        TORCH_CHECK(m_rows >= 1 && m_rows <= MaxRows, "zu_cw: m_rows out of range for this instantiation");
        TORCH_CHECK(m_rows <= M, "zu_cw: m_rows exceeds A rows");
        TORCH_CHECK(Mode == 4 || Mode == 5 || z.size(InKernelZ ? 1 : 0) == m_rows, "z rows must equal m_rows");
        const int R = (Mode == 4) ? static_cast<int>(z.size(1) / 2) : static_cast<int>(z.size(InKernelZ ? 2 : 1));   // Mode 4: Ztag [m_rows, 2R]
        TORCH_CHECK(R == StaticRank, "z rank mismatch for static-rank kernel");
        const int rank_words = (R + 31) / 32;
        TORCH_CHECK(U_col32.size(0) == N, "U_col32 N mismatch");
        TORCH_CHECK(U_col32.size(1) == rank_words, "U_col32 rank-word mismatch");
        const int tiles_n = (N + 63) / 64;
        const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        const int side_ctas = std::max(0, std::min(sm_count - tiles_n, tiles_n));
        const int side_kind = 0;                             // CUDA cores only (binary branch never touches tensor cores)
        if constexpr (InKernelZ) {
            TORCH_CHECK(X_bf16.scalar_type() == at::kBFloat16 && X_bf16.dim() == 2 && X_bf16.is_contiguous(), "X_bf16 must be bf16 2D contiguous");
            TORCH_CHECK(X_bf16.size(0) == m_rows && X_bf16.size(1) == K, "X_bf16 shape must be [m_rows, K]");
            TORCH_CHECK(K % 8 == 0, "zu_cw: K must be a multiple of 8");
            TORCH_CHECK(V_packed.scalar_type() == at::kByte && V_packed.dim() == 2 && V_packed.is_contiguous(), "V_packed must be uint8 2D contiguous");
            TORCH_CHECK(V_packed.size(0) == R && V_packed.size(1) == K / 8, "V_packed shape must be [R, K/8]");
            TORCH_CHECK(S.scalar_type() == at::kBFloat16 && S.numel() == R && S.is_contiguous(), "S must be bf16 [R]");
            TORCH_CHECK(tiles_n <= sm_count, "xvzu_cw: needs every N tile co-resident (N/64 <= SM count)");
            if (side_ctas == 0) {
                const int n_rg = (static_cast<int>(m_rows) + BinaryFetch::kRowBatch - 1) / BinaryFetch::kRowBatch;
                const int T_g = tiles_n / n_rg;
                TORCH_CHECK(T_g >= 1, "xvzu_cw: more row groups than tiles");
            }
            TORCH_CHECK(Flags.defined() && Flags.scalar_type() == at::kInt && Flags.numel() >= 1, "Flags[0] is the arrival counter (int32)");
            if (side_ctas > 0) {
                TORCH_CHECK(M <= 128, "xvzu_cw: single M tile only (tile grid must be (tiles_n, 1, 1))");
            }
        }

        auto D = torch::empty({L, M, N}, torch::dtype(element_traits<ElementD>::scalar_type).device(A.device()));
        {
            static bool printed_mode = false;
            if (!printed_mode && getenv("BILOCO_VERBOSE") != nullptr) {
                printed_mode = true;
                printf("[zu_cw Mode=%d R=%d MaxRows=%d] smem=%d stages=%d\n", Mode, StaticRank, MaxRows, static_cast<int>(Gemm::GemmKernel::SharedStorageSize), static_cast<int>(Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages));
            }
        }

        Gemm gemm;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, L});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});
        LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, L));
        LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, L));

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, L},
            {static_cast<typename ElementA::DataType const *>(A.data_ptr()), stride_A,
             static_cast<typename ElementB::DataType const *>(B.data_ptr()), stride_B,
             static_cast<typename ElementA::ScaleFactorType const *>(A_sf.data_ptr()), layout_SFA,
             static_cast<typename ElementB::ScaleFactorType const *>(B_sf.data_ptr()), layout_SFB},
            {{},
             nullptr,
             stride_C,
             static_cast<ElementD *>(D.data_ptr()),
             stride_D}};

        typename BinaryFetch::Arguments bargs;
        bargs.z_ptr = z.data_ptr<ElementCompute>();
        bargs.u_col32_ptr = U_col32.data_ptr<int32_t>();
        bargs.rank = R;
        bargs.rank_words = rank_words;
        bargs.n_cols = N;
        bargs.m_rows = static_cast<int>(m_rows);
        bargs.pdl_wait = pdl ? 1 : 0;
        if constexpr (InKernelZ) {
            bargs.x_ptr = reinterpret_cast<__nv_bfloat16 const*>(X_bf16.data_ptr());
            bargs.v_ptr = V_packed.data_ptr<uint8_t>();
            bargs.s_ptr = reinterpret_cast<__nv_bfloat16 const*>(S.data_ptr());
            bargs.k_cols = K;
            bargs.flags_ptr = Flags.data_ptr<int32_t>();
            bargs.num_side = side_ctas;
            bargs.side_kind = side_kind;
            int side_rc = 8, side_groups = 1, n_units_k0 = 0;
            TORCH_CHECK(side_ctas > 0, "xvzu_cw: the side path needs side CTAs (N/64 < SM count)");
            if (MaxRows > 1) TORCH_CHECK(N % 64 == 0, "xvzu_cw (M > 1): N must be a multiple of 64 (U tile bulk copy)");
            if (side_kind == 0) {
                if constexpr (Mode == 6) {
                    TORCH_CHECK(V_perm.defined() && V_perm.scalar_type() == at::kBFloat16 && V_perm.numel() == static_cast<int64_t>(R) * K && V_perm.is_contiguous(), "Mode 6 (dense): V_perm must be dense V bf16 [R, K]");
                    TORCH_CHECK(U_rw.defined() && U_rw.scalar_type() == at::kBFloat16 && U_rw.numel() == static_cast<int64_t>(R) * N && U_rw.is_contiguous(), "Mode 6 (dense): U_rw must be dense U bf16 [R, N]");
                    TORCH_CHECK(m_rows == 1 && static_cast<int>(m_rows) * R <= side_ctas, "Mode 6 (dense): M = 1 and one side CTA per rank");
                } else {
                TORCH_CHECK(V_perm.defined() && V_perm.scalar_type() == at::kByte && V_perm.numel() == static_cast<int64_t>(R) * 128 * 8, "kind 0 needs V_perm [R][128] x 8 bytes (nibble-interleaved)");
                }
                TORCH_CHECK(K == 5120, "kind 0 side units assume K = 5120 (128 threads x 10 nibbles)");
                constexpr int kSmem = static_cast<int>(Gemm::GemmKernel::SharedStorageSize);
                // ranks per unit: minimise (units per CTA) x (LUT build + per-rank cost) in issue cycles
                double best = 1e30;
                for (int rc : {8, 16, 32}) {
                    if (R % rc != 0 || rc > BinaryFetch::kRcMax) continue;
                    if (BinaryFetch::side_group_bytes(rc) > kSmem) continue;
                    const int units = static_cast<int>(m_rows) * (R / rc);
                    const double per_unit = (rc <= 16) ? (130.0 * rc) : (900.0 + 95.0 * rc);   // FFMA-only vs LUT build + per-rank (cycles)
                    const double cost = static_cast<double>((units + std::max(1, side_ctas) - 1) / std::max(1, side_ctas)) * per_unit;
                    if (cost < best) { best = cost; side_rc = rc; }
                }
                TORCH_CHECK(R % side_rc == 0 && side_rc >= 8 && side_rc <= BinaryFetch::kRcMax && side_rc % 8 == 0, "bad side_rc");
                TORCH_CHECK(BinaryFetch::side_group_bytes(side_rc) <= kSmem, "side unit smem exceeds the kernel smem");
                n_units_k0 = static_cast<int>(m_rows) * (R / side_rc);
                bargs.vperm_ptr = reinterpret_cast<uint8_t const*>(V_perm.data_ptr());
                bargs.side_rc = side_rc;
                bargs.side_mode = (side_rc <= 16) ? 1 : 0;
                bargs.side_groups = side_groups;
                bargs.z_copies = 8;
                bargs.dbg_flags = 0;
                bargs.m2_warm = 0;

                if constexpr (Mode == 6) {
                    bargs.u_rw_ptr = reinterpret_cast<int32_t const*>(U_rw.data_ptr());
                } else if (U_rw.defined() && U_rw.numel() > 0) {
                    TORCH_CHECK(U_rw.scalar_type() == at::kInt && U_rw.dim() == 2 && U_rw.size(0) == rank_words && U_rw.size(1) == N && U_rw.is_contiguous(), "U_rw must be int32 [rank_words][N] contiguous");
                    bargs.u_rw_ptr = U_rw.data_ptr<int32_t>();
                }
                static bool printed = false;
                if (!printed && getenv("BILOCO_VERBOSE") != nullptr) {
                    printed = true;
                    printf("[xvzu_cw] R=%d M=%d side_ctas=%d rc=%d units=%d dbg=%d smem=%d stages=%d\n", R, static_cast<int>(m_rows), side_ctas, side_rc, n_units_k0, bargs.dbg_flags, kSmem, static_cast<int>(Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages));
                }
            }
            bargs.n_units = side_ctas > 0 ? n_units_k0 : tiles_n;
            bargs.side_ks = 1;
            if (Diag.defined() && Diag.numel() > 0) {
                TORCH_CHECK(Diag.scalar_type() == at::kLong && Diag.numel() >= 4096, "Diag must be int64 [>= 4096]");
                bargs.diag_ptr = Diag.data_ptr<int64_t>();
            }
            bargs.num_workers = (N + 63) / 64;
            bargs.epoch = static_cast<int>(epoch);
        }
        if constexpr (Mode == 5) {
            // one launch: all CTAs run the LUT branch at entry (bsvd_branch_lut.cuh), tiles then run the GEMM
            TORCH_CHECK(m_rows >= 1 && m_rows <= 64 && m_rows <= MaxRows, "xvzu_lut: 1 <= m_rows <= 64");
            TORCH_CHECK(R % 128 == 0 && N % 64 == 0 && (K / 4) % bsvd_br::kSub == 0, "xvzu_lut: R % 128, N % 64, K % 256");
            TORCH_CHECK(X_bf16.scalar_type() == at::kBFloat16 && X_bf16.is_contiguous() && X_bf16.numel() == m_rows * K, "X_bf16 must be bf16 [m_rows, K]");
            TORCH_CHECK(V_packed.scalar_type() == at::kByte && V_packed.dim() == 2 && V_packed.size(0) == R && V_packed.size(1) == K / 8 && V_packed.is_contiguous(), "V_packed [R, K/8]");
            TORCH_CHECK(S.scalar_type() == at::kBFloat16 && S.numel() == R, "S bf16 [R]");
            const int Mp = (static_cast<int>(m_rows) + 15) / 16 * 16;
            TORCH_CHECK(z.scalar_type() == at::kFloat && z.numel() >= 2 * Mp * R, "Zbuf fp32 [2 * Mp, R] (zero-initialised)");
            TORCH_CHECK(Corr.defined() && Corr.scalar_type() == at::kFloat && Corr.numel() >= static_cast<int64_t>(N / 64) * Mp * 64, "Corr fp32 [N/64, Mp, 64]");
            TORCH_CHECK(Flags.defined() && Flags.scalar_type() == at::kInt && Flags.numel() >= 1 + N / 64, "Cnt int32 [1 + N/64] (zero-initialised)");
            TORCH_CHECK(side_ctas > 0 && M <= 128, "xvzu_lut: needs side CTAs and a single M tile");
            constexpr int kSmem = static_cast<int>(Gemm::GemmKernel::SharedStorageSize);
            TORCH_CHECK(bsvd_br::smem_bytes(R) <= kSmem, "xvzu_lut: branch smem exceeds the kernel smem");
            bsvd_br::Args& br = bargs.br;
            br.x = reinterpret_cast<__nv_bfloat16 const*>(X_bf16.data_ptr());
            br.v = V_packed.data_ptr<uint8_t>();
            br.s = reinterpret_cast<__nv_bfloat16 const*>(S.data_ptr());
            br.ucol = U_col32.data_ptr<int32_t>();
            br.z = z.data_ptr<float>();
            br.corr = Corr.data_ptr<float>();
            br.cnt = Flags.data_ptr<int32_t>();
            br.M = static_cast<int>(m_rows); br.Mp = Mp; br.K = K; br.N = N; br.R = R; br.epoch = static_cast<int>(epoch);
            // K groups per z item: largest divisor of the 64-nibble sub-chunk count with <= 128 z items
            const int nsub = (K / 4) / bsvd_br::kSub;
            int kg = 1;
            for (int d = 1; d <= nsub; ++d) {
                if (nsub % d == 0 && (Mp / 16) * (R / 32) * d <= 128) kg = d;
            }
            while (kg < nsub && ((K / 4) / kg > bsvd_br::kMaxNibItem || nsub % kg != 0)) ++kg;   // staged item fits in smem
            TORCH_CHECK(kg >= 1 && nsub % kg == 0, "xvzu_lut: bad kg");
            br.kg = kg;
            bargs.num_side = side_ctas;
            bargs.br_tiles = tiles_n;
            bargs.br_tiles_work = 1;
            static bool printed5 = false;
            if (!printed5 && getenv("BILOCO_VERBOSE") != nullptr) {
                printed5 = true;
                printf("[xvzu_lut] R=%d M=%d Mp=%d side=%d tiles=%d kg=%d smem=%d (branch %d) stages=%d\n", R, static_cast<int>(m_rows), Mp, side_ctas, tiles_n, kg, kSmem, bsvd_br::smem_bytes(R), static_cast<int>(Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages));
            }
        }
        if constexpr (Mode == 4) {
            // one launch: side CTAs (extra grid layer) compute complete S-scaled z ranks and publish them tagged;
            // tiles run the external-z path with a tagged poll in place of the z load
            TORCH_CHECK(MaxRows <= 8 || StaticRank <= 128, "xvzu_m4 (M = 16): R <= 128 (z staged in the correction buffer)");
            const bool rows_mode = (MaxRows == 2 || MaxRows == 4 || MaxRows == 8);   // small M: one side CTA per rank block, all rows
            TORCH_CHECK(X_bf16.scalar_type() == at::kBFloat16 && X_bf16.is_contiguous() && X_bf16.numel() == m_rows * K, "X_bf16 must be bf16 [m_rows, K]");
            TORCH_CHECK(V_packed.scalar_type() == at::kByte && V_packed.dim() == 2 && V_packed.is_contiguous() && V_packed.size(0) == R && V_packed.size(1) == K / 8, "V_packed must be uint8 [R, K/8]");
            TORCH_CHECK(S.scalar_type() == at::kBFloat16 && S.numel() == R && S.is_contiguous(), "S must be bf16 [R]");
            TORCH_CHECK(M <= 128, "xvzu_m4: single M tile (tile grid (tiles_n, 1, 1))");
            const bool direct_default = (m_rows == 1);
            // M = 1 default (bf16-direct side worker): ranks per side CTA = clamp(R / 32, 2, 8) (sweep 2026-09-27:
            // fewer side CTAs lower the fixed cost, more ranks per CTA delay z; r16 2, r128 4, r512 8)
            int zw_rc = direct_default ? std::max(2, std::min(8, R / 32)) : 8;
            if (zw_rc > R) zw_rc = R;
            while (zw_rc < 32 && (static_cast<int>(m_rows) * (R / zw_rc) > side_ctas || (!direct_default && zw_rc < 8))) zw_rc *= 2;
            if (MaxRows == 16 && R % 32 == 0) zw_rc = 32;   // 16-row kernel: the LUT side worker needs 32-rank units at K = 5120
            const int Nw = K / 4, Q = 256 / zw_rc;
            TORCH_CHECK(zw_rc == 1 || zw_rc == 2 || zw_rc == 4 || zw_rc == 8 || zw_rc == 16 || zw_rc == 32, "xvzu_m4: zw_rc must be a power of two <= 32");
            // small M side worker: 3 = the M = 1 bf16-direct worker per (row, rank block) when m_rows * R / rc fits
            // in the side CTAs with rc <= 8; otherwise one CTA per rank block for all rows: 6 = rank-inner direct
            // (R / 64 ranks per CTA), 5 = direct (V nibble shared by the rows), 4 = multi-row LUT.
            // Default (sweep 2026-09-27, K = N = 5120): 3 for M > 2 when it fits (M 3-4 at r128), else 6.
            int rows_side = 5;
            if (rows_mode) {
                int rc_direct = 1;
                while (rc_direct < 8 && static_cast<int>(m_rows) * (R / rc_direct) > side_ctas) rc_direct *= 2;
                const bool direct_fits = static_cast<int>(m_rows) * (R / rc_direct) <= side_ctas;
                rows_side = (MaxRows == 4 && m_rows > 2 && direct_fits) ? 3 : 6;   // the 2- and 8-row kernels compile only 6
                if (rows_side == 3) {
                    TORCH_CHECK(direct_fits, "xvzu_m4: per-row direct side worker needs m_rows * R / 8 <= side CTAs");
                    zw_rc = rc_direct;
                } else if (rows_side == 6) {
                    zw_rc = std::min(8, std::max(1, R / 64));   // fixed by the instantiation (kRc6)
                } else {
                    zw_rc = std::max(1, R / 64);             // 64 side CTAs (r128: 2 ranks each, r512: 8)
                }
                TORCH_CHECK(R % zw_rc == 0 && zw_rc <= 8 && K % 1024 == 0, "xvzu_m4 rows mode: bad rank block");
                TORCH_CHECK((rows_side != 3 ? R / zw_rc : static_cast<int>(m_rows) * (R / zw_rc)) <= side_ctas, "xvzu_m4 rows mode: too many side units");
                TORCH_CHECK(rows_side != 5 || 8 % zw_rc == 0, "xvzu_m4 rows mode 5: zw_rc must divide 8");
                TORCH_CHECK(!(MaxRows == 2 && rows_side == 4), "xvzu_m4: the multi-row LUT side worker needs the 4- or 8-row kernel");
                TORCH_CHECK(!(MaxRows == 2 || MaxRows == 8) || rows_side == 6, "xvzu_m4: the 2- and 8-row kernels compile only the rank-inner side worker (rows_side 6)");
            }
            const bool rows_block = rows_mode && (rows_side == 4 || rows_side == 5 || rows_side == 6);   // one side CTA per rank block (all rows)
            TORCH_CHECK(R % zw_rc == 0 && (rows_block || static_cast<int>(m_rows) * (R / zw_rc) <= side_ctas), "xvzu_m4: m_rows x R / zw_rc units exceed the side CTAs (", side_ctas, ")");
            (void)Q;
            constexpr int kSmem = static_cast<int>(Gemm::GemmKernel::SharedStorageSize);
            TORCH_CHECK(K % 64 == 0, "xvzu_m4: K must be a multiple of 64");
            bargs.x_ptr = reinterpret_cast<__nv_bfloat16 const*>(X_bf16.data_ptr());
            bargs.v_ptr = V_packed.data_ptr<uint8_t>();
            bargs.s_ptr = reinterpret_cast<__nv_bfloat16 const*>(S.data_ptr());
            bargs.k_cols = K;
            bargs.num_side = side_ctas;
            // launch only as many side CTAs as there are z units (idle side CTAs still cost ~0.01x)
            if (bargs.num_side > 0) {
                bargs.num_side = std::min(side_ctas, rows_block ? R / zw_rc : static_cast<int>(m_rows) * (R / zw_rc));
            }
            bargs.zw_rc = zw_rc;
            bargs.poll_gap_ns = 250;
            if (MaxRows <= 8) {
                TORCH_CHECK(z.size(0) % m_rows == 0 && z.size(0) / m_rows >= 1 && z.size(0) / m_rows <= 16, "xvzu_m4: Ztag rows must be z_copies x m_rows (1..16 copies)");
            }
            if (rows_block) {
                const int side_smem = rows_side == 4 ? bsvd_br::zrows_smem<(MaxRows == 8 ? 8 : 4)>(K, zw_rc)
                                    : rows_side == 6 ? bsvd_br::zinner_smem(K, zw_rc, MaxRows) : bsvd_br::zdirect_smem(K, zw_rc, MaxRows);
                TORCH_CHECK(side_smem <= kSmem, "xvzu_m4 rows mode: side smem");
            }
            bargs.z_copies = (z.size(0) % m_rows == 0) ? static_cast<int>(std::min<int64_t>(16, std::max<int64_t>(1, z.size(0) / m_rows))) : 1;
            bargs.tile_lut = 1;
            bargs.side_corr = 0;
            bargs.exp2 = 1;
            bargs.z4b = 0;
            TORCH_CHECK(!((MaxRows == 2 || MaxRows == 8) && bargs.z4b), "xvzu_m4: the 2- and 8-row kernels compile only the 8-byte tagged z poll");
            bargs.exp2r = 1;
            bargs.side_rot = 0;
            // 16-entry 4-row expansion: default at r >= 512 only (M 4: r512 9.75 -> 9.49 us; r128/r256 slower, the 64-entry build costs more)
            bargs.exp4r16 = (R >= 512 ? 1 : 0);
            bargs.poll_delay_ns = 0;
            bargs.poll_anchor_ns = ((MaxRows == 8 && R <= 128) ? 2300 : 0);
            bargs.poll_anchor_gap_ns = 150;
            // tile helper warps: required by the helper kernels (their epilogue expansion covers only 2/3 of the ranks)
            bargs.helpers = Sm90BinaryZUCwFetch<float, StaticRank, 64, MaxRows, Mode>::kHelp ? 1 : 0;
            // packed fp32x2 side loop (rank-inner worker, M > 2): sweep 2026-09-27: M 8 r256 8.98 -> 8.16, r512 10.96 -> 10.39;
            // M 4 r256 7.09 -> 6.80, r512 9.93 -> 9.17 us
            bargs.side_f2 = (m_rows > 2 ? 1 : 0);
            if (MaxRows > 1) bargs.poll_gap_ns = 0;
            if (bargs.side_corr) {
                TORCH_CHECK(zw_rc == 32 && R % 32 == 0 && N <= 5120 && N % 64 == 0, "xvzu_m4 side_corr: 32-rank units, N <= 5120, N % 64 == 0");
                TORCH_CHECK(z.numel() >= static_cast<int64_t>(tiles_n) * (R / 32) * 16 * 64, "xvzu_m4 side_corr: Ztag must hold tiles x R/32 x 16 x 64 values");
                bargs.z_copies = 1;
            }
            // warm-up pass: M = 1 only (an M > 1 expansion during the mainloop runs ~2.5 us; two do not fit)
            // (M 5-8 at r128 the expansion is short enough to warm during the mainloop: 7.95 -> 6.93 us)
            // cheap warm pass (instruction fetch only): default on for the 2-row kernel (same build: M 2 r512 7.14 -> 6.95, r256 6.39 -> 6.29 us)
            bargs.warm_cheap = (MaxRows == 2) ? 1 : 0;
            // v2 2-row expansion: default on at R >= 512 (same build: M2 r512 7.19 -> 6.81 us; r256 6.38 -> 6.46, stays v1)
            // 2-row kernel: the expansion variant is compiled per rank (v2 at R >= 512 needs the cheap warm pass's U staging)
            bargs.exp2v2 = (MaxRows == 2 && R >= 512) ? 1 : 0;
            TORCH_CHECK(!(MaxRows == 2 && R >= 512) || bargs.warm_cheap, "xvzu_m4: the 2-row kernel at R >= 512 needs warm_cheap");
            bargs.tile_warm = ((m_rows == 1 || (m_rows > 4 && R <= 128)) ? 1 : 0);
            if (bargs.warm_cheap) bargs.tile_warm = 1;
            bargs.side_unroll = 5;
            bargs.zw_mode = rows_mode ? rows_side : ((m_rows == 1) ? 3 : 0);   // M = 1: bf16-direct side worker; small M: 3 or 4
            if (bargs.zw_mode == 0) {
                TORCH_CHECK(K * 2 + Nw * 20 * 4 <= kSmem, "xvzu_m4: side LUT exceeds the kernel smem");
                TORCH_CHECK((Nw / (256 / zw_rc)) % 32 == 0 && (Nw / (256 / zw_rc)) / 8 <= 20, "xvzu_m4: LUT side mode needs 16-byte aligned V slices (zw_rc = 32 at K = 5120)");
            }
            if (bargs.zw_mode == 3) {
                TORCH_CHECK(zw_rc <= 8 && K % 1024 == 0, "xvzu_m4: bf16-direct side mode needs zw_rc <= 8 and K % 1024 == 0");
                TORCH_CHECK(bsvd_br::zdirect_smem(K, zw_rc, 1) <= kSmem, "xvzu_m4: bf16-direct side worker smem");
            }
            if (bargs.zw_mode == 1) {
                TORCH_CHECK(zw_rc <= 32 && K % 1024 == 0 && zw_rc * (K / 8) + K * 2 + 32 * (K / 32 + 4) * 4 + 256 <= kSmem, "xvzu_m4: direct side mode needs zw_rc <= 32 and K % 1024 == 0");
            }
            bargs.epoch = static_cast<int>(epoch);
            bargs.dbg_flags = 0;
            if (Diag.defined() && Diag.numel() > 0) {
                TORCH_CHECK(Diag.scalar_type() == at::kLong && Diag.numel() >= 4096, "Diag must be int64 [>= 4096]");
                bargs.diag_ptr = Diag.data_ptr<int64_t>();
            }
            static bool printed4 = false;
            if (!printed4 && getenv("BILOCO_VERBOSE") != nullptr) {
                printed4 = true;
                printf("[xvzu_m4] helpers=%d ", bargs.helpers);
                printf("[xvzu_m4] R=%d M=%d side_ctas=%d num_side=%d zw_rc=%d zw_mode=%d units=%d gap=%d copies=%d tile_lut=%d warm=%d dbg=%d smem=%d stages=%d\n", R, static_cast<int>(m_rows), side_ctas, bargs.num_side, zw_rc, bargs.zw_mode, static_cast<int>(m_rows) * (R / zw_rc), bargs.poll_gap_ns, bargs.z_copies, bargs.tile_lut, bargs.tile_warm, bargs.dbg_flags, kSmem, static_cast<int>(Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages));
            }
        }
        args.epilogue.thread = {
            {{{1.0f}, {static_cast<ElementScalar const *>(alpha.data_ptr())}}, {}, {}},
            bargs,
            {}
        };
        if constexpr (InKernelZ || Mode == 4 || Mode == 5) {
            // side CTAs: AlongM appends them along y of the (1, tiles) grid; AlongN adds a y layer to the (tiles, 1) grid
            // default AlongM: the side CTAs are appended along y of the (1, tiles) grid (no surplus CTAs; ~0.02x
            // cheaper than the (tiles, 2) AlongN layer); 1 = AlongN, 2 = scheduler default
            const int rast = (bargs.num_side > 0 ? 0 : 2);
            if (rast != 2) {   // 2: leave the scheduler default (only valid without a side layer)
                args.scheduler.raster_order = (rast == 0)
                    ? decltype(args.scheduler.raster_order)::AlongM : decltype(args.scheduler.raster_order)::AlongN;
            }
            {
                using FP = typename CollectiveEpilogue::Params::FusionParamsT;
                FP probe = CollectiveEpilogue::FusionCallbacks::to_underlying_arguments(cute::make_shape(M, N, K, L), args.epilogue.thread, nullptr);
                TORCH_CHECK(cutlass::gemm::kernel::detail::bsvd_side_finder<FP>::num_side(probe) == bargs.num_side,
                            "BiLoCo side hook did not resolve num_side (got ", cutlass::gemm::kernel::detail::bsvd_side_finder<FP>::num_side(probe),
                            ") FP = ", c10::demangle(typeid(FP).name()));
            }
        }

        CUTLASS_CHECK(gemm.can_implement(args));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        auto workspace_size = Gemm::get_workspace_size(args);
        auto workspace = torch::empty(
            {static_cast<int64_t>(workspace_size)},
            torch::dtype(torch::kUInt8).device(A.device()));
        void* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;
        CUTLASS_CHECK(gemm.initialize(args, workspace_ptr, stream));
        CUTLASS_CHECK(gemm.run(stream, nullptr, pdl));
        return D;
    }

    // Mode 4 (M=1): one launch. Side CTAs compute S-scaled z = S * (V x) on CUDA cores and publish it as
    // (value, epoch) pairs in Ztag [1, 2R]; tiles poll Ztag, expand with the U LUT, and add the correction.
    torch::Tensor gemm_fp4fp4_binary_xvzu_m4_smalln_dispatch(torch::Tensor const &A,
                                                             torch::Tensor const &B,
                                                             torch::Tensor const &A_sf,
                                                             torch::Tensor const &B_sf,
                                                             torch::Tensor const &alpha,
                                                             torch::Tensor const &X_bf16,
                                                             torch::Tensor const &V_packed,
                                                             torch::Tensor const &S,
                                                             torch::Tensor const &U_col32,
                                                             torch::Tensor const &Ztag,
                                                             torch::Tensor const &Diag,
                                                             int64_t m_rows,
                                                             int64_t epoch)
    {
        TORCH_CHECK(Ztag.dim() == 2 && Ztag.size(0) % m_rows == 0, "Ztag must be fp32 [z_copies * m_rows, 2R]");
        const int R = static_cast<int>(Ztag.size(1) / 2);
        const torch::Tensor none;
        // M = 2: the 2-row kernel at r >= 256 (r128: the 4-row kernel measured 6.30 vs 6.67 us); env overrides
        const bool m2_rows2 = R >= 256;
        if (m_rows == 2 && m2_rows2) {
            switch (R) {
            case 128: return gemm_fp4fp4_binary_zu_cw_smalln_impl<128, 2, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            case 256: return gemm_fp4fp4_binary_zu_cw_smalln_impl<256, 2, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            case 512: return gemm_fp4fp4_binary_zu_cw_smalln_impl<512, 2, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            default: TORCH_CHECK(false, "unsupported xvzu_m4 rank for M = 2: ", R);
            }
        }
        if (m_rows > 1 && m_rows <= 4) {
            switch (R) {
            case 64: return gemm_fp4fp4_binary_zu_cw_smalln_impl<64, 4, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            case 128: return gemm_fp4fp4_binary_zu_cw_smalln_impl<128, 4, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            case 256: return gemm_fp4fp4_binary_zu_cw_smalln_impl<256, 4, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            case 512: return gemm_fp4fp4_binary_zu_cw_smalln_impl<512, 4, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            default: TORCH_CHECK(false, "unsupported xvzu_m4 rank for M <= 4: ", R);
            }
        }
        if (m_rows > 4 && m_rows <= 8) {
            switch (R) {
            case 64: TORCH_CHECK(false, "mode 4: the 8-row r=64 kernel produces NaN outputs (disabled); use mode 2");
            case 128: return gemm_fp4fp4_binary_zu_cw_smalln_impl<128, 8, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            case 256: return gemm_fp4fp4_binary_zu_cw_smalln_impl<256, 8, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            case 512: return gemm_fp4fp4_binary_zu_cw_smalln_impl<512, 8, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            default: TORCH_CHECK(false, "unsupported xvzu_m4 rank for M <= 8: ", R);
            }
        }
        if (m_rows > 1) {
            switch (R) {
            case 64: return gemm_fp4fp4_binary_zu_cw_smalln_impl<64, 16, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            case 128: return gemm_fp4fp4_binary_zu_cw_smalln_impl<128, 16, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
            default:
                TORCH_CHECK(false, "unsupported xvzu_m4 rank for M > 1: ", R);
            }
        }
        switch (R) {
        case 16: return gemm_fp4fp4_binary_zu_cw_smalln_impl<16, 1, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
        case 32: return gemm_fp4fp4_binary_zu_cw_smalln_impl<32, 1, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
        case 64: return gemm_fp4fp4_binary_zu_cw_smalln_impl<64, 1, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
        case 128: return gemm_fp4fp4_binary_zu_cw_smalln_impl<128, 1, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
        case 256: return gemm_fp4fp4_binary_zu_cw_smalln_impl<256, 1, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
        case 512: return gemm_fp4fp4_binary_zu_cw_smalln_impl<512, 1, 4>(A, B, A_sf, B_sf, alpha, Ztag, U_col32, m_rows, X_bf16, V_packed, S, none, epoch, 0, 0, Diag);
        default:
            TORCH_CHECK(false, "unsupported xvzu_m4 rank: ", R);
        }
        return torch::Tensor();
    }

    // Mode 5 (1 <= M <= 64): one launch; every CTA runs the LUT branch (z, then the correction) at kernel
    // entry, then the tiles run the FP4 GEMM and add their correction slice in the epilogue.
    torch::Tensor gemm_fp4fp4_binary_xvzu_lut_smalln_dispatch(torch::Tensor const &A, torch::Tensor const &B,
                                                              torch::Tensor const &A_sf, torch::Tensor const &B_sf,
                                                              torch::Tensor const &alpha, torch::Tensor const &X_bf16,
                                                              torch::Tensor const &V_packed, torch::Tensor const &S,
                                                              torch::Tensor const &U_col32, torch::Tensor const &Zbuf,
                                                              torch::Tensor const &Corr, torch::Tensor const &Cnt,
                                                              int64_t m_rows, int64_t epoch)
    {
        TORCH_CHECK(Zbuf.dim() == 2, "Zbuf must be fp32 [2 * Mp, R]");
        const int R = static_cast<int>(Zbuf.size(1));
        const torch::Tensor none;
        switch (R) {
        case 128: return gemm_fp4fp4_binary_zu_cw_smalln_impl<128, 64, 5>(A, B, A_sf, B_sf, alpha, Zbuf, U_col32, m_rows, X_bf16, V_packed, S, Cnt, epoch, 0, 0, none, 0, false, none, none, Corr);
        case 256: return gemm_fp4fp4_binary_zu_cw_smalln_impl<256, 64, 5>(A, B, A_sf, B_sf, alpha, Zbuf, U_col32, m_rows, X_bf16, V_packed, S, Cnt, epoch, 0, 0, none, 0, false, none, none, Corr);
        case 512: return gemm_fp4fp4_binary_zu_cw_smalln_impl<512, 64, 5>(A, B, A_sf, B_sf, alpha, Zbuf, U_col32, m_rows, X_bf16, V_packed, S, Cnt, epoch, 0, 0, none, 0, false, none, none, Corr);
        default: TORCH_CHECK(false, "xvzu_lut: unsupported rank ", R);
        }
        return torch::Tensor();
    }

#define BSVD_CL_SCHEDULER cutlass::gemm::StaticPersistentScheduler

    torch::Tensor gemm_fp4fp4_binary_xvzu_cw_smalln_dispatch(torch::Tensor const &A,
                                                             torch::Tensor const &B,
                                                             torch::Tensor const &A_sf,
                                                             torch::Tensor const &B_sf,
                                                             torch::Tensor const &alpha,
                                                             torch::Tensor const &X_bf16,
                                                             torch::Tensor const &V_packed,
                                                             torch::Tensor const &S,
                                                             torch::Tensor const &U_col32,
                                                             torch::Tensor const &Z_scratch,
                                                             torch::Tensor const &Flags,
                                                             torch::Tensor const &Diag,
                                                             torch::Tensor const &V_perm,
                                                             torch::Tensor const &U_rw,
                                                             int64_t m_rows,
                                                             int64_t epoch)
    {
        TORCH_CHECK(Z_scratch.dim() == 4, "Z_scratch must be [8, m_rows, R, 2]");
        const int R = Z_scratch.size(2);
#define BILOCO_MODE2_CASE(RANK) \
        case RANK: \
            if constexpr (RANK == 32) { if (m_rows == 1 && getenv("BILOCO_DENSE_BF16") != nullptr) return gemm_fp4fp4_binary_zu_cw_smalln_impl<RANK, 1, 6>(A, B, A_sf, B_sf, alpha, Z_scratch, U_col32, m_rows, X_bf16, V_packed, S, Flags, epoch, 0, 0, Diag, 0, false, V_perm, U_rw); } \
            if (m_rows == 1) return gemm_fp4fp4_binary_zu_cw_smalln_impl<RANK, 1, 2>(A, B, A_sf, B_sf, alpha, Z_scratch, U_col32, m_rows, X_bf16, V_packed, S, Flags, epoch, 0, 0, Diag, 0, false, V_perm, U_rw); \
            if constexpr (RANK <= 128) { if (m_rows > 16) return gemm_fp4fp4_binary_zu_cw_smalln_impl<RANK, 32, 2>(A, B, A_sf, B_sf, alpha, Z_scratch, U_col32, m_rows, X_bf16, V_packed, S, Flags, epoch, 0, 0, Diag, 0, false, V_perm, U_rw); } \
            return gemm_fp4fp4_binary_zu_cw_smalln_impl<RANK, 16, 2>(A, B, A_sf, B_sf, alpha, Z_scratch, U_col32, m_rows, X_bf16, V_packed, S, Flags, epoch, 0, 0, Diag, 0, false, V_perm, U_rw);
        switch (R) {
        BILOCO_MODE2_CASE(16)
        BILOCO_MODE2_CASE(32)
        BILOCO_MODE2_CASE(64)
        BILOCO_MODE2_CASE(128)
        BILOCO_MODE2_CASE(256)
        BILOCO_MODE2_CASE(512)
        default:
            TORCH_CHECK(false, "unsupported xvzu_cw rank: ", R);
        }
#undef BILOCO_MODE2_CASE
        return torch::Tensor();
    }

    TORCH_LIBRARY_IMPL(biloco, CUDA, m)
    {

        /* NVFP4 */
        m.impl("fp4_gemm",
               &gemm_fp4fp4_accum_fp32<
                   cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                   Shape<_256, _256, _256>,
                   Shape<_4, _1, _1>,
                   cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>);

        /* NVFP4 small-tile (1SM, 128x16x256) — for skinny GEMMs with small N
         * (e.g. low-rank K-contraction with N=R≤32). 2SM cluster + 256x256
         * tile rejects M<256/N<256 via can_implement; this 1SM variant runs
         * at decode shapes (M=1, N=R=8/16/32). N is padded/masked to the
         * 16-tile by CUTLASS so any N ≥ 1 works.
         */
        m.impl("fp4_gemm_n64",
               &gemm_fp4fp4_accum_fp32<
                   cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                   Shape<_128, _64, _256>,
                   Shape<_1, _1, _1>,
                   cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>);

        m.impl("fp4_gemm_n128",
               &gemm_fp4fp4_accum_fp32<
                   cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                   Shape<_128, _128, _256>,
                   Shape<_1, _1, _1>,
                   cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>);
        m.impl("biloco_gemm_mode2",
               &gemm_fp4fp4_binary_xvzu_cw_smalln_dispatch);
        m.impl("biloco_gemm_mode4",
               &gemm_fp4fp4_binary_xvzu_m4_smalln_dispatch);
        m.impl("biloco_gemm_mode5", &gemm_fp4fp4_binary_xvzu_lut_smalln_dispatch);

        /* NVFP4 main matmul WITH residual: D = alpha * A@B + 1.0 * C.
         * Used to fuse the binary-branch's final y_main + y_branch add into
         * fp4-main's epilogue, eliminating a separate kernel launch.
         */
        m.impl("fp4_gemm_residual",
               &gemm_fp4fp4_residual_accum_fp32<
                   cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                   Shape<_256, _256, _256>,
                   Shape<_4, _1, _1>,
                   cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>);
    }
}
