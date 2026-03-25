import time
import torch
import argparse
import functools
import numpy as np
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.custom_all_reduce import init_custom_ar


def init_world(device_id, num_devices, parts, port=24514):
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=device_id,
        world_size=num_devices,
        device_id=device_id,
    )
    group_size = num_devices // parts
    group_id = device_id // group_size
    group_ranks = list(range(group_id * group_size, (group_id + 1) * group_size))
    group = dist.new_group(ranks=group_ranks)
    print(f"[init_world] device_id:{device_id}, group_ranks:{group_ranks}", flush=True)
    return group


@dataclass
class Args:
    dtype: torch.dtype
    n: int
    num_devices: int
    parts: int
    nsamples: int


def create_inputs(args):
    group_size = args.num_devices // args.parts
    inputs = []
    for part in range(args.parts):
        for rank in range(group_size):
            device_id = part * group_size + rank
            for i in range(args.nsamples):
                x = torch.randn(args.n, dtype=args.dtype, device=f"cuda:{device_id}")
                inputs.append(x)
    return inputs


def create_outputs(args):
    group_size = args.num_devices // args.parts
    outputs = []
    for part in range(args.parts):
        for rank in range(group_size):
            device_id = part * group_size + rank
            for i in range(args.nsamples):
                x = torch.randn(args.n, dtype=args.dtype, device=f"cuda:{device_id}")
                outputs.append(x)
    return outputs


def ref_worker(device_id, num_devices, parts, nsamples, inputs, outputs):
    group = init_world(device_id, num_devices, parts)
    # rank = dist.get_rank(group=group)
    # world_size = dist.get_world_size(group=group)
    for i in range(nsamples):
        input = inputs[device_id * nsamples + i]
        output = outputs[device_id * nsamples + i]
        output.copy_(input) # use inplace allreduce
        dist.all_reduce(output, group=group)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    dist.destroy_process_group()


def ref_func(args, inputs, outputs):
    mp.spawn(
        ref_worker,
        args=(args.num_devices, args.parts, args.nsamples, inputs, outputs),
        nprocs=args.num_devices,
        join=True
    )


def golden_ref_func(args, inputs, outputs):
    group_size = args.num_devices // args.parts
    for part in range(args.parts):
        for i in range(args.nsamples):
            outputs_part = []
            for rank in range(group_size):
                device_id = part * group_size + rank
                outputs_part.append(inputs[device_id * args.nsamples + i].cpu())
            outputs_part = torch.stack(outputs_part, dim=-1).sum(dim=-1)
            for rank in range(group_size):
                device_id = part * group_size + rank
                outputs[device_id * args.nsamples + i] = outputs_part.cuda(device_id)


def worker(device_id, num_devices, parts, nsamples, inputs, outputs):
    group = init_world(device_id, num_devices, parts)
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    meta = torch.empty((0,), device=device_id, dtype=torch.int8)
    rank_data = inputs[device_id * nsamples]
    handles = [torch.empty((1,), device="cpu", dtype=torch.uint8) for _ in range(world_size)]
    offsets = [0 for _ in range(world_size)]
    fa = init_custom_ar(meta, rank_data, handles, offsets, rank=rank)
    for i in range(nsamples):
        input = inputs[device_id * nsamples + i]
        output = outputs[device_id * nsamples + i]
        fa.custom_all_reduce(input, open_fp8_quant=False, out=output)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    dist.destroy_process_group()


def func(args, inputs, outputs):
    mp.spawn(
        worker,
        args=(args.num_devices, args.parts, args.nsamples, inputs, outputs),
        nprocs=args.num_devices,
        join=True
    )


def benchmark(args, func, ref_func):
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    func(args, inputs, outputs)
    ref_func(args, inputs, ref_outputs)
    max_diff_global = float(-1)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output)
        # assert is_allclose == True
        maxdiff_out = (output - ref_output).abs().max().item()
        max_diff_global = max(max_diff_global, maxdiff_out)
    print(f"max_diff_global:{max_diff_global}")

    # get ref_func perf
    print("===================== [REF] =====================")
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        ref_func(args, inputs, ref_outputs)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)

    # get func perf
    print("===================== [FLYDSL] =====================")
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        func(args, inputs, outputs)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--num_devices", type=int, required=True)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--nsamples", type=int, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
    # rm -rf ~/.flydsl/ ; python3 allreduce.py --nsamples=10 --num_devices=4 --dtype=bf16 --n=16384
