# FlyDSL-Examples

This is an unofficial collection of FlyDSL GEMM kernel examples.
For AMD GPU users, the goal is to build peak-performance GEMM kernels from scratch while showing how to write FlyDSL operators step by step.
With https://github.com/rocm/flydsl, high-performance GPU kernels can be developed in Python for AMD GPUs, in a style similar to CUDA/CuteDSL.

This repository currently includes:

- [x] B/F16-GEMM-WMMA (for MI350)
- [x] FP8-PTPC-GEMM-WMMA (for MI350)

![HGEMM BF16 benchmark](images/hgemm_benchmark.svg)

## How to build install FlyDSL on AMD GPUs

Check the ROCm version using `amd-smi`. My version is `7.0.1`.

```bash
# fast-way: pip install flydsl
git clone https://github.com/ROCm/FlyDSL
cd FlyDSL
bash scripts/build_llvm.sh -j64
bash scripts/build.sh -j64
pip install -e .
```

## GEMM-WMMA Test

```bash
rm -rf ~/.flydsl/ ; pytest -sv test_hgemm.py
```

> Documents for IR study: https://mlir.llvm.org/docs/

> Contact: xytpai@gmail.com
