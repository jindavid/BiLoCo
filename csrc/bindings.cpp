#include <Python.h>
#include <torch/extension.h>

extern "C"
{
    PyObject *PyInit__C(void)
    {
        static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL, NULL, NULL, NULL, NULL};
        return PyModule_Create(&module_def);
    }
}

// Operator schemas. A / A_sf: NVFP4 activation codes [1, M, K/2] and block scales; B / B_sf: NVFP4 weight
// codes [N, K/2] and block scales; alpha: fp32 [1] global scale. Outputs are bf16 [1, M, N].
namespace biloco
{
    TORCH_LIBRARY(biloco, m)
    {
        // FP4-only GEMMs. fp4_gemm: CUTLASS 256x256 2-SM tile (the default a framework would pick).
        // fp4_gemm_n64 / fp4_gemm_n128: 1-SM 128x64 / 128x128 tiles. fp4_gemm_n64 with A zero-padded to
        // 128 rows is the FP4-only baseline of the paper.
        m.def("fp4_gemm(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
        m.def("fp4_gemm_n64(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
        m.def("fp4_gemm_n128(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
        // D = alpha * A @ B^T + C_residual (used by the unfused reference).
        m.def("fp4_gemm_residual(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha, Tensor C_residual) -> Tensor");

        // Fused BiLoCo kernels: FP4 GEMM (128x64 tiles, A padded to 128 rows) plus the binary low-rank term
        // sum_i S_i (x . v_i) u_i, computed from the packed signs in the same launch and added in the epilogue.
        // X_bf16 [M, K]: unquantized activation for the low-rank term; V_packed [R, K/8]; S bf16 [R];
        // U_col32 [N, R/32] int32 (see biloco/kernel.py for the layouts).
        //   mode2: z computed by side CTAs and tiles, epoch-flagged scratch (M <= 32, M = 32 only for R <= 128).
        //          With env BILOCO_DENSE_BF16=1 at M = 1, R = 32 it runs the dense BF16 rank-32 control instead
        //          (V_perm = dense V [32, K] bf16, U_rw = dense U [32, N] bf16).
        //   mode4: z computed by side CTAs only and published with epoch tags (M <= 16).
        //   mode5: every CTA computes the correction at kernel entry, then the GEMM runs (R in {128, 256, 512}).
        m.def("biloco_gemm_mode2(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha, Tensor X_bf16, "
              "Tensor V_packed, Tensor S, Tensor U_col32, Tensor Z_scratch, Tensor Flags, Tensor Diag, "
              "Tensor V_perm, Tensor U_rw, int m_rows, int epoch) -> Tensor");
        m.def("biloco_gemm_mode4(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha, Tensor X_bf16, "
              "Tensor V_packed, Tensor S, Tensor U_col32, Tensor Ztag, Tensor Diag, int m_rows, int epoch) -> Tensor");
        m.def("biloco_gemm_mode5(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha, Tensor X_bf16, "
              "Tensor V_packed, Tensor S, Tensor U_col32, Tensor Zbuf, Tensor Corr, Tensor Cnt, int m_rows, int epoch) -> Tensor");

        // Unfused reference for the low-rank term: z = S * (X sign(V)^T) (fp32), Y = z sign(U) (bf16).
        m.def("binary_z_reference(Tensor X, Tensor V_packed, Tensor S) -> Tensor");
        m.def("binary_expand_reference(Tensor z, Tensor U_packed, Tensor(a!) Y) -> ()");
    }
}
