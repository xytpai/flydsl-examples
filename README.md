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
python3 pointwise_add.py --n 10000000 --dtype f32
```

```txt
run: /home/yuxu/pointwise_add.py, args: Namespace(n=10000000, dtype='f32')
validation passed!

===================== [REF] =====================
[W224 07:00:55.984698439 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.369ms       100.00%       4.369ms      43.686us           100  
                                        hipLaunchKernel         9.44%     386.842us         9.44%     386.842us       3.868us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize        90.56%       3.710ms        90.56%       3.710ms       3.710ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.097ms
Self CUDA time total: 4.369ms

===================== [FLYDSL] =====================
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
     pointwise_add_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       3.891ms       100.00%       3.891ms      38.906us           100  
          hipStreamCreate        69.06%      78.745ms        69.06%      78.745ms     787.451us       0.000us         0.00%       0.000us       0.000us           100  
    hipModuleLaunchKernel         0.75%     854.424us         0.75%     854.424us       8.544us       0.000us         0.00%       0.000us       0.000us           100  
     hipStreamSynchronize         5.34%       6.093ms         5.34%       6.093ms      60.934us       0.000us         0.00%       0.000us       0.000us           100  
         hipStreamDestroy        24.77%      28.247ms        24.77%      28.247ms     282.472us       0.000us         0.00%       0.000us       0.000us           100  
     hipDeviceSynchronize         0.07%      80.387us         0.07%      80.387us       0.796us       0.000us         0.00%       0.000us       0.000us           101  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 114.020ms
Self CUDA time total: 3.891ms
```

---

> Contact: xytpai@gmail.com
