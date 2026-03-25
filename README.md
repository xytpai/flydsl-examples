# FlyDSL-Examples

This is an unofficial FlyDSL kernel example repository.
If you are using AMD GPUs, we hope to provide a step-by-step guide to help you get familiar with writing operators using FlyDSL.
Empowered by https://github.com/rocm/flydsl we are now able to develop high-performance GPU kernels in Python on AMD GPU, similar to CUDA/CuteDSL.
This repository will provide the following examples from scratch:

- [x] Pointwise Add
- [x] Batch Reduce
- [x] RMS Norm
- [x] HGEMM (wmma)
- [x] Allreduce
- [ ] Flash Attention
- [ ] Linear Attention (NEED REBASE)
- [ ] GEMM-FP8

For IR study: https://mlir.llvm.org/docs/

## 0. How to build install FlyDSL on AMD GPUs

Check the ROCm version using `amd-smi`. My version is `7.0.1`.

```bash
git clone https://github.com/ROCm/FlyDSL
cd FlyDSL
git checkout 76c924b4d25b3c18242535139583eeeee2708d08
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

Take a closer look at details, we just use CUDA-like coding style:

```python
index = bidx * BLOCK_WORK_SIZE + tidx * VEC_SIZE
remaining = n - index
if arith.cmpi(arith.CmpIPredicate.ult, remaining, fx.Int32(VEC_SIZE)):
    for i in range_constexpr(VEC_SIZE):
        if arith.cmpi(arith.CmpIPredicate.ult, index + i, fx.Int32(n)):
            C_[index + i] = A_[index + i] + B_[index + i]
else:
    vec_a = A_.vec_load((index,), VEC_SIZE)
    vec_b = B_.vec_load((index,), VEC_SIZE)
    vec_c = vec_a + vec_b
    C_.vec_store((index,), vec_c, VEC_SIZE)
```

## 2. Batch Reduce

```bash
python3 batch_reduce.py --batch_size=4 --reduce_size=2048 --dtype=f16
```

```txt
run: /home/yuxu/flydsl-examples/batch_reduce.py, args: Namespace(batch_size=4, reduce_size=2048, dtype='f16')
validation passed!

===================== [REF] =====================
[W317 16:34:50.229649829 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     580.540us       100.00%     580.540us       5.805us           100  
                                        hipLaunchKernel        92.08%     546.354us        92.08%     546.354us       5.464us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize         7.92%      47.021us         7.92%      47.021us      47.021us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 593.375us
Self CUDA time total: 580.540us

===================== [FLYDSL] =====================
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
          batch_reduce_kernel_0         0.00%       0.000us         0.00%       0.000us       0.000us     333.622us       100.00%     333.622us       3.336us           100  
    hipStreamCreateWithPriority        94.46%      17.358ms        94.46%      17.358ms       1.578ms       0.000us         0.00%       0.000us       0.000us            11  
          hipModuleLaunchKernel         2.98%     547.905us         2.98%     547.905us       5.479us       0.000us         0.00%       0.000us       0.000us           100  
           hipDeviceSynchronize         2.56%     470.132us         2.56%     470.132us     470.132us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 18.376ms
Self CUDA time total: 333.622us
```

Leverage tensor utilities to perform vectorized loads and stores, improving code readability:

```python
c_zero_f = arith.constant(0.0, type=T.f32)
init_state = [c_zero_f]
for vec_idx, state in range(tidx * VEC_SIZE, fx.Int32(reduce_size), fx.Int32(BLOCK_WORK_SIZE), init=init_state):
    x_sum = state[0]
    x_vec = X_.vec_load((bidx, vec_idx), VEC_SIZE)
    x_vec = x_vec.extf(acc_vec_t)
    x_sum = x_sum + vector.ReductionOp(T.f32, vector.CombiningKind.ADD, x_vec).dest
    results = yield [x_sum]

for offset in WARP_SIZE_SHFL_OFFSETS:
    results = results + results.shuffle_xor(fx.Int32(offset), fx.Int32(WARP_SIZE))
        
base_ptr = allocator.get_base()
smem_ptr = SmemPtr(base_ptr, smem_offset, T.f32, shape=(NUM_WARPS,))
smem_ = STensor(smem_ptr, T.f32, shape=(NUM_WARPS,))
smem_[wid] = results
gpu.barrier()

if arith.cmpi(arith.CmpIPredicate.eq, tidx, fx.Int32(0)):
    sum_x = c_zero_f
    for i in range_constexpr(NUM_WARPS):
        sum_x = sum_x + smem_[i]
    Y_[bidx] = sum_x.truncf(dtype_)
```

## 3. RMS Norm

```bash
python3 rms_norm.py --batch_size=16 --norm_size=4096 --dtype=f16
```

```txt
run: /home/yuxu/flydsl-examples/rms_norm.py, args: Namespace(batch_size=16, norm_size=4096, dtype='f16')
validation passed!

===================== [REF] =====================
[W317 17:24:20.785488429 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.198ms        21.88%       1.198ms      11.976us           100  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     798.265us        14.58%     798.265us       7.983us           100  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us     634.341us        11.59%     634.341us       6.343us           100  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     563.704us        10.30%     563.704us       5.637us           100  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     507.583us         9.27%     507.583us       5.076us           100  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     503.183us         9.19%     503.183us       5.032us           100  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us     486.582us         8.89%     486.582us       4.866us           100  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     437.222us         7.99%     437.222us       4.372us           100  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     345.622us         6.31%     345.622us       3.456us           100  
                                        hipLaunchKernel        86.16%       2.937ms        86.16%       2.937ms       3.671us       0.000us         0.00%       0.000us       0.000us           800  
                                         hipMemcpyAsync        12.46%     424.586us        12.46%     424.586us       4.246us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize         1.38%      47.052us         1.38%      47.052us      47.052us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.409ms
Self CUDA time total: 5.474ms

===================== [FLYDSL] =====================
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
               rmsnorm_kernel_0         0.00%       0.000us         0.00%       0.000us       0.000us     447.382us       100.00%     447.382us       4.474us           100  
    hipStreamCreateWithPriority        94.77%      18.165ms        94.77%      18.165ms       1.651ms       0.000us         0.00%       0.000us       0.000us            11  
          hipModuleLaunchKernel         2.88%     551.394us         2.88%     551.394us       5.514us       0.000us         0.00%       0.000us       0.000us           100  
           hipDeviceSynchronize         2.35%     450.961us         2.35%     450.961us     450.961us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 19.168ms
Self CUDA time total: 447.382us
```

## 4. HGEMM

```bash
python3 hgemm.py --m=4096 --n=4096 --k=4096 --dtype=f16
```

```txt
run: /home/yuxu/flydsl-examples/hgemm.py, args: Namespace(m=4096, n=4096, k=4096, dtype='f16')
maxdiff_out:0.0625
maxdiff_out:0.0625
maxdiff_out:0.0625
maxdiff_out:0.0625
maxdiff_out:0.0625
===================== [REF] =====================
[W325 08:58:44.666889393 collection.cpp:1116] Warning: ROCTracer produced duplicate flow start: 4 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Cijk_Alik_Bljk_HHS_BH_UserArgs_MT256x224x64_MI16x16x...         0.00%       0.000us         0.00%       0.000us       0.000us      73.170ms       100.00%      73.170ms     731.697us           100  
                            hipGetDevicePropertiesR0600         0.13%      96.362us         0.13%      96.362us       0.321us       0.000us         0.00%       0.000us       0.000us           300  
                               hipExtModuleLaunchKernel         0.57%     408.387us         0.57%     408.387us       4.084us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize        99.30%      71.121ms        99.30%      71.121ms      71.121ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 71.626ms
Self CUDA time total: 73.170ms

===================== [FLYDSL] =====================
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         hgemm_f16_128x128x64_S1TN_BP_0         0.00%       0.000us         0.00%       0.000us       0.000us      73.178ms        90.42%      73.178ms     731.779us           100  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us       7.751ms         9.58%       7.751ms      77.512us           100  
                                        hipLaunchKernel         0.66%     448.365us         0.66%     448.365us       4.484us       0.000us         0.00%       0.000us       0.000us           100  
                                  hipModuleLaunchKernel         0.63%     427.903us         0.63%     427.903us       4.279us       0.000us         0.00%       0.000us       0.000us           100  
                                   hipDeviceSynchronize        98.70%      66.616ms        98.70%      66.616ms      66.616ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 67.492ms
Self CUDA time total: 80.929ms
```

## 5. Allreduce

```bash
python3 allreduce.py --nsamples=10 --num_devices=4 --dtype=bf16 --n=16384
```

```txt
run: /home/yuxu/flydsl-examples/allreduce.py, args: Namespace(n=16384, dtype='bf16', num_devices=4, parts=1, nsamples=10)

max_diff_global:0.0625

===================== [REF] =====================
[init_world] device_id:0, group_ranks:[0, 1, 2, 3]
[init_world] device_id:1, group_ranks:[0, 1, 2, 3]
[init_world] device_id:2, group_ranks:[0, 1, 2, 3]
[init_world] device_id:3, group_ranks:[0, 1, 2, 3]
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
      hipIpcGetMemHandle        64.20%     529.977us        64.20%     529.977us       1.656us           320  
    hipStreamSynchronize        33.94%     280.182us        33.94%     280.182us       0.876us           320  
    hipDeviceSynchronize         1.86%      15.379us         1.86%      15.379us      15.379us             1  
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 825.538us

===================== [FLYDSL] =====================
[init_world] device_id:0, group_ranks:[0, 1, 2, 3]
[init_world] device_id:3, group_ranks:[0, 1, 2, 3]
[init_world] device_id:1, group_ranks:[0, 1, 2, 3]
[init_world] device_id:2, group_ranks:[0, 1, 2, 3]
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
      hipIpcGetMemHandle        73.38%     494.781us        73.38%     494.781us       1.546us           320  
    hipStreamSynchronize        23.94%     161.413us        23.94%     161.413us       0.504us           320  
    hipDeviceSynchronize         2.68%      18.090us         2.68%      18.090us      18.090us             1  
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 674.284us
```

---

> Contact: xytpai@gmail.com
