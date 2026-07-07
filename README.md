# FlyDSL-Examples

This is an unofficial FlyDSL kernel example repository.
If you are using AMD GPUs, we hope to provide a step-by-step guide to help you get familiar with writing operators using FlyDSL.
Empowered by https://github.com/rocm/flydsl we are now able to develop high-performance GPU kernels in Python on AMD GPU, similar to CUDA/CuteDSL.
This repository will provide the following examples from scratch:

- [x] B/F16-GEMM-WMMA (for MI350)
- [x] FP8-PTPC-GEMM-WMMA (for MI350)

For IR study: https://mlir.llvm.org/docs/

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

### HGEMM BF16 Benchmark

![HGEMM BF16 benchmark](images/hgemm_benchmark.svg)

---

> Contact: xytpai@gmail.com
