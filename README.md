# FlyDSL-Examples

This is an unofficial FlyDSL kernel example repository.
If you are using AMD GPUs, we hope to provide a step-by-step guide to help you get familiar with writing operators using FlyDSL.
Empowered by https://github.com/rocm/flydsl we are now able to develop high-performance GEMM based operators in Python on AMD GPU, similar to CuteDSL.
This repository will provide the following examples from scratch:

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

# After this you will see the installed path.
# -- Installing: /home/yuxu/llvm-project/mlir_install/lib/cmake/llvm/LLVMConfigExtensions.cmake
# Creating tarball...
# ==============================================
# LLVM/MLIR build completed successfully!

export MLIR_PATH=/home/yuxu/llvm-project/build
export MLIR_DIR=/home/yuxu/llvm-project
./flir/build.sh
python3 -m pip install -e .
```

To check whether the package works:

```bash
bash scripts/run_tests.sh
```

---

> Contact: xytpai@gmail.com
