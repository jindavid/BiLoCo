# BiLoCo: Binary Low-Rank Corrections for Post-Training Quantization

BiLoCo splits each weight matrix into an NVFP4 residual and a low-rank component whose factors are binary
sign vectors:

    W ≈ U diag(α) Vᵀ + R,    U ∈ {±1}^{m×r},  V ∈ {±1}^{n×r},  R stored in NVFP4

A binary rank-one term costs 16× less storage than a BF16 one, so the storage of SVDQuant's BF16 rank-32
component buys binary rank 512. The binary factors turn the low-rank product into additions and
subtractions, which a fused Blackwell kernel computes on CUDA cores in the same launch as the FP4
tensor-core GEMM.

This repository contains the method (decomposition and whole-model PTQ, run as fake quantization for
the accuracy evaluation) and the fused B300 decode kernel.

## Layout

```
biloco/
  peel.py                greedy sign peeling and the joint coefficient refit
  layer.py               BiLoCoLinear: smoothing, decomposition, NVFP4 residual, fake-quant forward
  model.py               whole-model PTQ with SmoothQuant calibration, save/load
  fp4.py                 NVFP4 fake quantization ("fouroversix")
  data.py, evaluate.py   datasets and evaluation
  kernel.py              fused kernel: operand packing, launcher, unfused reference, timing
csrc/              CUDA extension
  biloco_gemm.cu         FP4 GEMMs and the fused kernel
  biloco_side.cuh        side-CTA z worker and device helpers
  biloco_branch_lut.cuh  LUT branch for the all-CTA schedule
  reference_ops.cu       unfused reference
patches/           CUTLASS hooks for the side CTAs
scripts/           eval_ppl.py; kernel_table.py, kernel_shapes.py, equal_rank.py (kernel timing)
tests/
```

## Install

```bash
git clone --recurse-submodules https://github.com/jindavid/BiLoCo.git && cd BiLoCo
pip install -r requirements.txt
```

The kernel needs a Blackwell B300 (sm_103a; set
`BILOCO_CUDA_ARCH=100a` for B200) and CUDA 13:

```bash
git -C third_party/cutlass apply ../../patches/cutlass_biloco.patch
pip install -e . --no-build-isolation
```

## Decomposition

```python
from biloco import decompose, set_peel_seed
set_peel_seed(42)
U, V, alpha, R = decompose(W.float(), r=512, T=5)   # W ≈ U diag(alpha) Vᵀ + R; quantize R to NVFP4
```

Each rank-one term starts from random signs and alternates `u = sign(R v)`, `v = sign(Rᵀ u)` for T steps,
takes the Frobenius-optimal coefficient and deflates the residual. After all r terms, the coefficients are
refit jointly by least squares for the fixed signs.


## Kernel

The fused kernel computes `y = FP4(x) FP4(R)ᵀ + Σᵢ αᵢ (x·vᵢ) uᵢ` in one launch for decode batches of up to
32.

- The FP4 GEMM runs on tensor cores with 128×64 tiles, one CTA per output tile. The activation is
  zero-padded to 128 rows.
- Side CTAs on the remaining SMs compute z = α ⊙ (V x) on CUDA cores from the bit-packed signs of V.
- The epilogue warps of each GEMM CTA expand z through the bit-packed signs of U and add the result to the
  accumulators before the BF16 store.

`BiLoCoGemm` picks the fastest mode for the batch size and rank
(`biloco.kernel.DEFAULT_MODE`), and `biloco.kernel.modes_for(M, r, K)` lists the ones available.

| Mode | How z reaches the GEMM tiles | Main use |
|---|---|---|
| 2 | side CTAs write z to scratch and signal an arrival counter | small batches at K = 5120 |
| 4 | side CTAs publish each z value with an epoch tag that the tiles poll | small batches, also other shapes |
| 5 | every CTA computes the correction at kernel entry, before the GEMM | large batch × rank |

```python
from biloco.kernel import BiLoCoGemm, pack_signs

# U [N, r], V [K, r] and alpha [r] come from biloco.decompose
# W_fp4, W_sf, w_global_scale: NVFP4 codes, block scales and global scale of the residual R
gemm = BiLoCoGemm(W_fp4, W_sf, w_global_scale, pack_signs(V.T), pack_signs(U.T), alpha.bfloat16(), M=1)
y = gemm(A_pad, A_sf, x)   # A_pad: NVFP4 activation zero-padded to 128 rows, x: BF16 activation [M, K]
                           # y: [1, 128, N] BF16, rows >= M are padding
```

## Performance

`biloco.quantize_model` replaces every `nn.Linear` (including `lm_head`), applies SmoothQuant scales
collected on a BF16 copy of the model, decomposes each smoothed weight, and fake-quantizes the residual
and the activations to NVFP4.

```bash
python scripts/eval_ppl.py --model_name Qwen/Qwen3-4B --method biloco   # BiLoCo
python scripts/eval_ppl.py --model_name Qwen/Qwen3-4B --method fp4      # NVFP4-only baseline
python scripts/eval_ppl.py --model_name Qwen/Qwen3-4B --method bf16     # BF16 reference
```

## License

MIT (see `LICENSE`). CUTLASS is a submodule under its own license.
