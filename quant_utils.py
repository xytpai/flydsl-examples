import torch
from flydsl.runtime.device import get_rocm_arch


def get_torch_fp8_dtype():
    arch = str(get_rocm_arch())
    if ("gfx95" in arch or "gfx12" in arch) and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    raise RuntimeError("This PyTorch build does not expose an E4M3 FP8 dtype")


fp8_dtype = get_torch_fp8_dtype()
fp8_max = float(torch.finfo(fp8_dtype).max)


def quantize_ptpc_fp8(x: torch.Tensor):
    scale = x.float().abs().amax(dim=1, keepdim=True) / fp8_max
    scale[scale == 0] = 1
    x_fp8 = (x.float() / scale).to(fp8_dtype)
    scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).squeeze(1)
    return x_fp8, scale.float()


def quantize_input_block_fp8(x: torch.Tensor, block_k: int = 128):
    rows, k = x.shape
    assert k % block_k == 0
    x_blocks = x.float().view(rows, k // block_k, block_k)
    scale = x_blocks.abs().amax(dim=2, keepdim=True) / fp8_max
    scale[scale == 0] = 1
    scale = scale.clamp_min(1e-30)
    x_fp8 = (x_blocks / scale).view(rows, k).to(fp8_dtype)
    scale = torch.nan_to_num(scale.squeeze(2), nan=1.0, posinf=1.0, neginf=1.0)
    return x_fp8, scale.float()


def quantize_weight_block_fp8(x: torch.Tensor, block_n: int = 128, block_k: int = 128):
    rows, k = x.shape
    assert rows % block_n == 0
    assert k % block_k == 0
    x_blocks = x.float().view(rows // block_n, block_n, k // block_k, block_k)
    scale = x_blocks.abs().amax(dim=(1, 3), keepdim=True) / fp8_max
    scale[scale == 0] = 1
    scale = scale.clamp_min(1e-30)
    x_fp8 = (x_blocks / scale).view(rows, k).to(fp8_dtype)
    scale = torch.nan_to_num(
        scale.squeeze(3).squeeze(1), nan=1.0, posinf=1.0, neginf=1.0
    )
    return x_fp8, scale.float()
