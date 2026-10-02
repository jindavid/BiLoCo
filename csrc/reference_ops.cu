// Unfused reference for the BiLoCo low-rank term, used to check the fused kernel:
//   binary_z_reference:      z[m, i] = S[i] * sum_k sign(V[i, k]) * X[m, k]   (fp32)
//   binary_expand_reference: Y[m, n] = sum_i z[m, i] * sign(U[i, n])          (bf16)
// Signs are bit-packed LSB-first (bit set = +1): V_packed [R, K/8], U_packed [R, N/8].
// Both use sum_k s_k x_k = 2 * sum_{s_k = +1} x_k - sum_k x_k.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace biloco {

namespace {

// Y = z * sign(U). Each CTA owns BLOCK_N outputs of one row and loops over the rank
// in chunks of K_TILE, staging the z chunk and the U slice in shared memory.
template <int K_TILE, int NUM_THREADS, int BLOCK_N, int MIN_CTAS = 1>
__global__ __launch_bounds__(NUM_THREADS, MIN_CTAS)
void expand_kernel(
    const float*         __restrict__ z,    // (M, R) fp32
    const uint8_t*       __restrict__ Up,   // (R, N/8) packed bits
    __nv_bfloat16*       __restrict__ Y,    // (M, N) bf16
    int N, int R)
{
    static_assert(BLOCK_N % 8 == 0, "BLOCK_N multiple of 8");
    const int n_block = blockIdx.x;
    const int m       = blockIdx.y;
    const int n_start = n_block * BLOCK_N;
    if (n_start >= N) return;

    const int tid = threadIdx.x;
    const int N_bytes = N / 8;

    constexpr int OUTPUTS_PER_THREAD = (BLOCK_N + NUM_THREADS - 1) / NUM_THREADS;
    float sum_plus[OUTPUTS_PER_THREAD];
    #pragma unroll
    for (int o = 0; o < OUTPUTS_PER_THREAD; ++o) sum_plus[o] = 0.0f;
    float sum_z_total = 0.0f;

    __shared__ float   z_smem[K_TILE];
    __shared__ uint8_t u_smem[K_TILE * (BLOCK_N / 8)];
    constexpr int u_n_bytes = BLOCK_N / 8;

    for (int r_off = 0; r_off < R; r_off += K_TILE) {
        for (int idx = tid; idx < K_TILE; idx += NUM_THREADS) {
            z_smem[idx] = z[m * R + r_off + idx];
        }
        for (int idx = tid; idx < K_TILE * u_n_bytes; idx += NUM_THREADS) {
            const int r = idx / u_n_bytes;
            const int b = idx % u_n_bytes;
            const int n_byte_global = n_start / 8 + b;
            u_smem[idx] = (n_byte_global < N_bytes) ? Up[(r_off + r) * N_bytes + n_byte_global] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < K_TILE; ++r) sum_z_total += z_smem[r];

        #pragma unroll
        for (int o = 0; o < OUTPUTS_PER_THREAD; ++o) {
            const int t = o * NUM_THREADS + tid;
            if (t >= BLOCK_N) break;
            const int n_byte_in_block = t / 8;
            const int bit_in_byte    = t % 8;
            #pragma unroll
            for (int r = 0; r < K_TILE; ++r) {
                const uint8_t ubyte = u_smem[r * u_n_bytes + n_byte_in_block];
                if ((ubyte >> bit_in_byte) & 1) sum_plus[o] += z_smem[r];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int o = 0; o < OUTPUTS_PER_THREAD; ++o) {
        const int t = o * NUM_THREADS + tid;
        if (t >= BLOCK_N) break;
        const int n_global = n_start + t;
        if (n_global >= N) break;
        Y[m * N + n_global] = __float2bfloat16(2.0f * sum_plus[o] - sum_z_total);
    }
}

// z = S * (X sign(V)^T). Grid (1, M, R / RANK_GROUP); each CTA writes RANK_GROUP values of one row.
template <int RANK_GROUP, int NUM_THREADS, int MIN_CTAS = 1>
__global__ __launch_bounds__(NUM_THREADS, MIN_CTAS)
void z_kernel(
    const __nv_bfloat16* __restrict__ X,    // (M, K)
    const uint8_t*       __restrict__ Vp,   // (R_total, K/8)
    const __nv_bfloat16* __restrict__ S,    // (R_total,)
    float*               __restrict__ z,    // (M, R_total) fp32
    int K)
{
    static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS multiple of warp size");
    constexpr int N_WARPS = NUM_THREADS / 32;

    const int m = blockIdx.y;
    const int g = blockIdx.z;
    const int my_rank_off = g * RANK_GROUP;
    const int R_total = gridDim.z * RANK_GROUP;

    const int tid = threadIdx.x;
    const int K_bytes = K / 8;

    float sum_total = 0.0f;
    float sum_plus[RANK_GROUP];
    #pragma unroll
    for (int r = 0; r < RANK_GROUP; ++r) sum_plus[r] = 0.0f;

    for (int k_byte = tid; k_byte < K_bytes; k_byte += NUM_THREADS) {
        const int k_start = k_byte * 8;
        const uint4 x_pack = *reinterpret_cast<const uint4*>(X + m * K + k_start);
        __nv_bfloat16 x[8];
        *reinterpret_cast<uint4*>(&x[0]) = x_pack;
        float xf[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) xf[i] = __bfloat162float(x[i]);
        #pragma unroll
        for (int i = 0; i < 8; ++i) sum_total += xf[i];
        #pragma unroll
        for (int r = 0; r < RANK_GROUP; ++r) {
            const uint8_t vbyte = Vp[(my_rank_off + r) * K_bytes + k_byte];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if ((vbyte >> i) & 1) sum_plus[r] += xf[i];
            }
        }
    }

    float acc[RANK_GROUP];
    #pragma unroll
    for (int r = 0; r < RANK_GROUP; ++r) acc[r] = 2.0f * sum_plus[r] - sum_total;

    #pragma unroll
    for (int r = 0; r < RANK_GROUP; ++r) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
        }
    }

    __shared__ float inter[RANK_GROUP * N_WARPS];
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < RANK_GROUP; ++r) inter[r * N_WARPS + warp] = acc[r];
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int r = 0; r < RANK_GROUP; ++r) {
            float v = (lane < N_WARPS) ? inter[r * N_WARPS + lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            }
            if (lane == 0) {
                z[m * R_total + my_rank_off + r] = v * __bfloat162float(S[my_rank_off + r]);
            }
        }
    }
}

torch::Tensor binary_z_reference(const torch::Tensor& X,
                                 const torch::Tensor& V_packed,
                                 const torch::Tensor& S)
{
    TORCH_CHECK(X.dim() == 2 && X.is_contiguous(),               "X must be 2D contiguous");
    TORCH_CHECK(V_packed.dim() == 2 && V_packed.is_contiguous(), "V_packed must be 2D contiguous");
    TORCH_CHECK(S.dim() == 1 && S.is_contiguous(),               "S must be 1D contiguous");
    TORCH_CHECK(X.scalar_type() == at::kBFloat16,                "X must be bf16");
    TORCH_CHECK(V_packed.scalar_type() == at::kByte,             "V_packed must be uint8");
    TORCH_CHECK(S.scalar_type() == at::kBFloat16,                "S must be bf16");
    TORCH_CHECK(X.is_cuda() && V_packed.is_cuda() && S.is_cuda(), "all tensors must be CUDA");

    const int M = X.size(0);
    const int K = X.size(1);
    const int R = V_packed.size(0);
    TORCH_CHECK(K % 8 == 0, "K must be multiple of 8");
    TORCH_CHECK(R == S.size(0), "rank mismatch S vs V");
    TORCH_CHECK(V_packed.size(1) == K / 8, "V_packed K-bytes mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto z = torch::empty({M, R}, X.options().dtype(at::kFloat));

    auto launch_with = [&](auto rank_group_v, auto nthr_v, auto min_ctas_v) {
        constexpr int RG = decltype(rank_group_v)::value;
        constexpr int NTHR = decltype(nthr_v)::value;
        constexpr int MIN_CTAS = decltype(min_ctas_v)::value;
        TORCH_CHECK(R % RG == 0, "rank must be a multiple of ", RG);
        dim3 grid(1, M, R / RG);
        z_kernel<RG, NTHR, MIN_CTAS><<<grid, NTHR, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
            V_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(S.data_ptr()),
            z.data_ptr<float>(),
            K);
    };

    if (R >= 512) {
        launch_with(std::integral_constant<int, 8>{}, std::integral_constant<int, 256>{}, std::integral_constant<int, 4>{});
    } else if (R >= 256) {
        launch_with(std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{}, std::integral_constant<int, 4>{});
    } else if (R == 16 || R == 32) {
        launch_with(std::integral_constant<int, 4>{}, std::integral_constant<int, 1024>{}, std::integral_constant<int, 1>{});
    } else {
        launch_with(std::integral_constant<int, 8>{}, std::integral_constant<int, 1024>{}, std::integral_constant<int, 1>{});
    }
    return z;
}

void binary_expand_reference(const torch::Tensor& z,
                             const torch::Tensor& U_packed,
                             torch::Tensor Y)
{
    TORCH_CHECK(z.dim() == 2 && z.is_contiguous(), "z must be 2D contiguous");
    TORCH_CHECK(U_packed.dim() == 2 && U_packed.is_contiguous(), "U_packed must be 2D contiguous");
    TORCH_CHECK(Y.dim() == 2 && Y.is_contiguous(), "Y must be 2D contiguous");
    TORCH_CHECK(z.scalar_type() == at::kFloat, "z must be fp32");
    TORCH_CHECK(U_packed.scalar_type() == at::kByte, "U_packed must be uint8");
    TORCH_CHECK(Y.scalar_type() == at::kBFloat16, "Y must be bf16");

    const int M = z.size(0);
    const int R = z.size(1);
    const int N = Y.size(1);
    TORCH_CHECK(N % 8 == 0, "N must be multiple of 8");
    TORCH_CHECK(U_packed.size(0) == R, "U rank mismatch");
    TORCH_CHECK(U_packed.size(1) == N / 8, "U N-bytes mismatch");
    TORCH_CHECK(Y.size(0) == M, "Y M mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto launch_with = [&](auto k_tile_v) {
        constexpr int KT = decltype(k_tile_v)::value;
        TORCH_CHECK(R % KT == 0, "rank must be a multiple of ", KT);
        dim3 grid((N + 127) / 128, M);
        expand_kernel<KT, 128, 128, 4><<<grid, 128, 0, stream>>>(
            z.data_ptr<float>(),
            U_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()),
            N, R);
    };
    if (R >= 64) {
        launch_with(std::integral_constant<int, 16>{});
    } else {
        launch_with(std::integral_constant<int, 8>{});
    }
}

}  // anonymous namespace

TORCH_LIBRARY_IMPL(biloco, CUDA, m)
{
    m.impl("binary_z_reference", &binary_z_reference);
    m.impl("binary_expand_reference", &binary_expand_reference);
}

}  // namespace biloco
