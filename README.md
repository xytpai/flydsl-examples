# FlyDSL-Examples

This is an unofficial FlyDSL kernel example repository.
If you are using AMD GPUs, we hope to provide a step-by-step guide to help you get familiar with writing operators using FlyDSL.
Empowered by https://github.com/rocm/flydsl we are now able to develop high-performance GEMM based operators in Python on AMD GPU, similar to CuteDSL.
This repository will provide the following examples from scratch:

- [x] Pointwise Add
- [ ] Reduction
- [ ] RMS Norm
- [ ] SGEMM
- [ ] HGEMM
- [ ] GEMM-FP8
- [ ] GEMM Fusions

## 0. How to install FlyDSL on AMD GPUs

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

## 1. Pointwise Add

```bash
python3 pointwise_add.py --n 16384000 --dtype f32
```

```txt
run: /home/yuxu/pointwise_add.py, args: Namespace(n=16384000, dtype='f32')
validation passed!
[W224 06:06:00.296290124 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     688.273us       100.00%     688.273us      68.827us            10  
                                        hipLaunchKernel        11.01%      69.075us        11.01%      69.075us       6.908us       0.000us         0.00%       0.000us       0.000us            10  
                                   hipDeviceSynchronize        88.99%     558.365us        88.99%     558.365us     558.365us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 627.440us
Self CUDA time total: 688.273us

[FlyDSL] elapsed_per_iter:1433.7459564208984 us
```

---

> Contact: xytpai@gmail.com
