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

for idx_in_vec in range(VEC_SIZE):
    idx_in_vec = flir.const_index(idx_in_vec)
    # Get a value
    a_val = frgA[(idx_in_vec, )]

```

---

> Contact: xytpai@gmail.com
