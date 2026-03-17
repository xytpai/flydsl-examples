# FlyDSL-Examples

This is an unofficial FlyDSL kernel example repository.
If you are using AMD GPUs, we hope to provide a step-by-step guide to help you get familiar with writing operators using FlyDSL.
Empowered by https://github.com/rocm/flydsl we are now able to develop high-performance GEMM based operators in Python on AMD GPU, similar to CuteDSL.
This repository will provide the following examples from scratch:

- [x] Pointwise Add
- [x] Batch Reduce
- [x] RMS Norm
- [x] HGEMM
- [ ] GEMM-FP8
- [ ] GEMM Fusions

For IR study: https://mlir.llvm.org/docs/

## 0. How to build install FlyDSL on AMD GPUs

Check the ROCm version using `amd-smi`. My version is `7.0.1`.

```bash
git clone https://github.com/ROCm/FlyDSL
cd FlyDSL
git checkout 429e6c7f82de4d2bb9a3013946617be2b1a1c791
bash scripts/build_llvm.sh -j64

# After this you will see the installed path.
# ==============================================
# LLVM/MLIR build completed successfully!

# To configure flydsl, use:
# cmake .. -DMLIR_DIR=/home/yuxu/llvm-project/build-flydsl/lib/cmake/mlir

# Packaged install prefix:
#   /home/yuxu/llvm-project/mlir_install
# Use with:
#   export MLIR_PATH=/home/yuxu/llvm-project/mlir_install
# Tarball:
#   /home/yuxu/llvm-project/mlir_install.tgz
# ==============================================

export MLIR_PATH=/home/yuxu/llvm-project/mlir_install
bash scripts/build.sh -j64
python3 -m pip install -e .
```

To check whether the package works:

```bash
bash scripts/run_tests.sh
```

To clean flydsl cache

```bash
rm -rf ~/.flydsl/
```

## 1. Pointwise Add

```bash
python3 pointwise_add.py --n 10000000 --dtype f32
```

```txt
run: /home/yuxu/flydsl-examples/pointwise_add.py, args: Namespace(n=10000000, dtype='f32')
validation passed!

===================== [REF] =====================
[W317 14:28:48.993059503 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.138ms       100.00%       4.138ms      41.379us           100  
                                        hipLaunchKernel         9.68%     373.103us         9.68%     373.103us       3.731us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize        90.32%       3.481ms        90.32%       3.481ms       3.481ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.855ms
Self CUDA time total: 4.138ms

===================== [FLYDSL] =====================
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
         pointwise_add_kernel_0         0.00%       0.000us         0.00%       0.000us       0.000us       3.697ms       100.00%       3.697ms      36.975us           100  
    hipStreamCreateWithPriority        77.58%      16.016ms        77.58%      16.016ms       1.456ms       0.000us         0.00%       0.000us       0.000us            11  
          hipModuleLaunchKernel         2.51%     518.197us         2.51%     518.197us       5.182us       0.000us         0.00%       0.000us       0.000us           100  
           hipDeviceSynchronize        19.91%       4.109ms        19.91%       4.109ms      40.685us       0.000us         0.00%       0.000us       0.000us           101  
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 20.644ms
Self CUDA time total: 3.697ms
```

Take a closer look at details, we just use CUDA like coding style:

```python
index = bidx * BLOCK_WORK_SIZE + tidx * VEC_SIZE_
remaining = n_ - index
if arith.cmpi(arith.CmpIPredicate.ult, remaining, VEC_SIZE_):
    for i in range_constexpr(VEC_SIZE_):
        if arith.cmpi(arith.CmpIPredicate.ult, index + i, n_):
            C_[index + i] = A_[index + i] + B_[index + i]
    else:
        vec_a = A_.vec_load((index,), VEC_SIZE_)
        vec_b = B_.vec_load((index,), VEC_SIZE_)
        vec_c = vec_a + vec_b
        C_.vec_store((index,), vec_c, VEC_SIZE_)
```

## 2. Batch Reduce

```bash
python3 batch_reduce.py --batch_size=4 --reduce_size=2048 --dtype=f16
```

```txt
run: /home/yuxu/flydsl-examples/batch_reduce.py, args: Namespace(batch_size=4, reduce_size=2048, dtype='f16')
validation passed!

===================== [REF] =====================
[W225 05:28:18.762480600 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     687.062us       100.00%     687.062us       6.871us           100  
                                        hipLaunchKernel        95.26%     460.725us        95.26%     460.725us       4.607us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize         4.74%      22.922us         4.74%      22.922us      22.922us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 483.647us
Self CUDA time total: 687.062us

===================== [FLYDSL] =====================
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
      batch_reduce_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     516.783us       100.00%     516.783us       5.168us           100  
          hipStreamCreate        72.63%      88.464ms        72.63%      88.464ms     884.637us       0.000us         0.00%       0.000us       0.000us           100  
    hipModuleLaunchKernel         0.84%       1.025ms         0.84%       1.025ms      10.254us       0.000us         0.00%       0.000us       0.000us           100  
     hipStreamSynchronize         2.17%       2.639ms         2.17%       2.639ms      26.388us       0.000us         0.00%       0.000us       0.000us           100  
         hipStreamDestroy        24.24%      29.527ms        24.24%      29.527ms     295.267us       0.000us         0.00%       0.000us       0.000us           100  
     hipDeviceSynchronize         0.11%     138.746us         0.11%     138.746us       1.374us       0.000us         0.00%       0.000us       0.000us           101  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 121.793ms
Self CUDA time total: 516.783us
```

Leverage vector utilities to perform vectorized loads and stores, improving code readability:

```python
@flir.kernel
def batch_reduce_kernel(
    self: flir.T.i64,
    X: lambda: T.memref(S, dtype.get()),
    Y: lambda: T.memref(S, dtype.get()),
    batch_size: lambda: T.index(),
    reduce_size: lambda: T.index(),
):
    tid_x = flir.thread_idx("x")
    bid_x = flir.block_idx("x")
    vec_type = VectorType.get([VEC_SIZE], self.dtype)
    acc_vec_type = VectorType.get([VEC_SIZE], self.acc_type)
    c_zero = arith.constant(0.0, type=self.acc_type)
    thread_sum = (c_zero)
    for vec_idx in range(tid_x * VEC_SIZE, reduce_size, BLOCK_WORK_SIZE):
        vec_addr = bid_x * reduce_size + vec_idx
        vec = flir.vector.load(vec_type, X, [arith.as_value(vec_addr)], alignment=16)
        vec = flir.arith.extf(acc_vec_type, arith.as_value(vec))
        red = flir.vector.reduction(self.acc_type, "add", arith.as_value(vec), fastmath=fm_fast)
        thread_sum = thread_sum + red
    block_reduce_add = make_block_reduce_add(tid_x, WARP_SIZE, RED_SLOTS)
    base_ptr = allocator.get_base()
    sum_val = block_reduce_add(thread_sum, self.smem(base_ptr).get())
    sum_val = flir.arith.truncf(self.dtype, (sum_val))
    flir.memref.store(arith.as_value(sum_val), Y, [flir.const_index(bid_x),])
```

## 3. RMS Norm

```bash
python3 rms_norm.py --batch_size=16 --norm_size=4096 --dtype=f16
```

```txt
run: /home/yuxu/flydsl-examples/rms_norm.py, args: Namespace(batch_size=16, norm_size=4096, dtype='f16')
validation passed!

===================== [REF] =====================
[W225 07:43:14.141697179 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.148ms        21.16%       1.148ms      11.479us           100  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     804.501us        14.83%     804.501us       8.045us           100  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us     630.502us        11.62%     630.502us       6.305us           100  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     547.743us        10.10%     547.743us       5.477us           100  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     508.705us         9.38%     508.705us       5.087us           100  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     506.545us         9.34%     506.545us       5.065us           100  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us     498.064us         9.18%     498.064us       4.981us           100  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     431.584us         7.96%     431.584us       4.316us           100  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     348.464us         6.42%     348.464us       3.485us           100  
                                        hipLaunchKernel        86.86%       2.928ms        86.86%       2.928ms       3.660us       0.000us         0.00%       0.000us       0.000us           800  
                                         hipMemcpyAsync        12.51%     421.693us        12.51%     421.693us       4.217us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize         0.64%      21.434us         0.64%      21.434us      21.434us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.371ms
Self CUDA time total: 5.424ms

===================== [FLYDSL] =====================
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
          rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     613.066us       100.00%     613.066us       6.131us           100  
          hipStreamCreate        68.49%      71.702ms        68.49%      71.702ms     717.024us       0.000us         0.00%       0.000us       0.000us           100  
    hipModuleLaunchKernel         0.78%     821.251us         0.78%     821.251us       8.213us       0.000us         0.00%       0.000us       0.000us           100  
     hipStreamSynchronize         3.02%       3.162ms         3.02%       3.162ms      31.625us       0.000us         0.00%       0.000us       0.000us           100  
         hipStreamDestroy        27.64%      28.939ms        27.64%      28.939ms     289.392us       0.000us         0.00%       0.000us       0.000us           100  
     hipDeviceSynchronize         0.07%      69.542us         0.07%      69.542us       0.689us       0.000us         0.00%       0.000us       0.000us           101  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 104.695ms
Self CUDA time total: 613.066us
```

## 4. HGEMM

```bash
python3 hgemm.py --m=8192 --n=8192 --k=8192 --dtype=f16
```

```txt
run: /home/yuxu/flydsl-examples/hgemm.py, args: Namespace(m=8192, n=8192, k=8192, dtype='f16')
validation passed!

===================== [REF] =====================
[W227 12:33:28.617539313 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 4 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Cijk_Alik_Bljk_HHS_BH_UserArgs_MT256x224x64_MI16x16x...         0.00%       0.000us         0.00%       0.000us       0.000us     531.365ms       100.00%     531.365ms       5.314ms           100  
                            hipGetDevicePropertiesR0600         0.02%     103.111us         0.02%     103.111us       0.344us       0.000us         0.00%       0.000us       0.000us           300  
                               hipExtModuleLaunchKernel         0.10%     529.805us         0.10%     529.805us       5.298us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize        99.88%     531.702ms        99.88%     531.702ms       5.264ms       0.000us         0.00%       0.000us       0.000us           101  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 532.335ms
Self CUDA time total: 531.365ms

===================== [FLYDSL] =====================
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           hgemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     576.349ms        95.32%     576.349ms       5.763ms           100  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us      28.318ms         4.68%      28.318ms     283.183us           100  
                                        hipLaunchKernel         0.12%     846.279us         0.12%     846.279us       8.463us       0.000us         0.00%       0.000us       0.000us           100  
                                        hipStreamCreate        10.32%      70.744ms        10.32%      70.744ms     707.441us       0.000us         0.00%       0.000us       0.000us           100  
                                  hipModuleLaunchKernel         0.13%     891.809us         0.13%     891.809us       8.918us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipStreamSynchronize        84.43%     578.859ms        84.43%     578.859ms       5.789ms       0.000us         0.00%       0.000us       0.000us           100  
                                       hipStreamDestroy         4.63%      31.726ms         4.63%      31.726ms     317.264us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize         0.37%       2.506ms         0.37%       2.506ms      24.816us       0.000us         0.00%       0.000us       0.000us           101  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 685.574ms
Self CUDA time total: 604.668ms
```

---

> Contact: xytpai@gmail.com
