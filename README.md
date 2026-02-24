# FlyDSL-Examples

This is an unofficial FlyDSL kernel example repository.
If you are using AMD GPUs, we hope to provide a step-by-step guide to help you get familiar with writing operators using FlyDSL.
Empowered by https://github.com/rocm/flydsl we are now able to develop high-performance GEMM based operators in Python on AMD GPU, similar to CuteDSL.
This repository will provide the followings examples:

- [ ] Pointwise Add
- [ ] Reduction
- [ ] RMS Norm
- [ ] SGEMM
- [ ] HGEMM
- [ ] GEMM-FP8
- [ ] GEMM Fusions

## 1. How to install FlyDSL on AMD GPUs

Check the ROCm version using `amd-smi`. My version is `7.0.1`.

```bash
git clone https://github.com/ROCm/FlyDSL
cd FlyDSL
bash scripts/build_llvm.sh
./flir/build.sh
python3 -m pip install -e .
```
