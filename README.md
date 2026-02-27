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

Let's take a closer look at details:

- Thread/Value Layout

```python
# To create a TV layout.
# thread_layout determines how are threads arranged in a thread block
thread_layout = flir.make_ordered_layout((BLOCK_SIZE,), order=(0,))
# value_layout determines how are values arranged in a thread
value_layout = flir.make_ordered_layout((VEC_SIZE,), order=(0,))
```

- Get the data for a single thread

```python
# define vectorized atomic load
copy_atom_load = flir.make_copy_atom(dtype.get(), vector_size=VEC_SIZE)
# define tv layout copy
tiled_copy_A = flir.make_tiled_copy_tv(copy_atom_load, thread_layout, value_layout, thr_shape=(BLOCK_SIZE,), val_shape=(VEC_SIZE,))

# Create a tensor view from a memref with a specific layout and shape.
tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
# Partition a tensor view.
gA = flir.zipped_divide(tensor_A, (BLOCK_WORK_SIZE,))
# Get tensor view of this block
blkA = gA[(bid_x,)]
# Get per-thread slice of the tiled copy
thr_copy_A = tiled_copy_A.get_slice(tid_x)
# Get tensor view of this thread
thrA = thr_copy_A.partition_S(blkA)

# Create register in this thread for A fragment
frgA = flir.make_fragment_like(thrA, dtype.get())

val_shape = tiled_copy_A.val_shape
# Create tensor in register for mask
frgPred = flir.make_rmem_tensor(val_shape, IntegerType.get_signless(1))
for idx_in_vec in range(val_shape[0]): # iter VEC_SIZE
    idx_in_vec = flir.const_index(idx_in_vec)

    # Return absolute coordinates for a given linear index.
    # thrCrd is just an identity tensor maps each coordinate to itself, useful for tracking
    # coordinates during partitioning.
    coords = thrCrd.coords_from_linear(idx_in_vec) 

    pred_val = flir.elem_less(coords, (n,))
    pred_offsets = tuple(frgPred.offsets_from_linear(idx_in_vec))
    frgPred[pred_offsets] = pred_val

# Copy to register
flir.copy(tiled_copy_A, thrA, frgA, pred=frgPred)

for idx_in_vec in range_constexpr(VEC_SIZE):
    idx_in_vec = flir.const_index(idx_in_vec)
    # Get a value
    a_val = frgA[(idx_in_vec, )]

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
