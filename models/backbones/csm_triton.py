# csm_triton.py — Enhanced with OSSM-style 8-direction scan (horizontal, vertical, diagonal, anti-diagonal)
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
from functools import lru_cache

# ============================
# Index Map Generators (Cached)
# ============================

@lru_cache(maxsize=128)
def _get_scan_indices(H: int, W: int, device_type: str) -> Tuple[torch.Tensor, ...]:
    device = torch.device(device_type)
    L = H * W
    indices = []

    # 0: left → right (row-major)
    idx0 = torch.arange(L, device=device)

    # 1: top → bottom (col-major) → equivalent to transpose flatten
    idx1 = torch.arange(L, device=device).view(H, W).t().flatten()

    # 2: right → left (flip row)
    idx2 = torch.arange(L, device=device).view(H, W).flip(dims=[1]).flatten()

    # 3: bottom → top (flip col)
    idx3 = torch.arange(L, device=device).view(H, W).flip(dims=[0]).t().flatten()

    # 4: main diagonal ↘
    mat = torch.arange(L, device=device).view(H, W)
    diag_idx = []
    for s in range(H + W - 1):
        for i in range(max(0, s - W + 1), min(H, s + 1)):
            j = s - i
            if 0 <= j < W:
                diag_idx.append(mat[i, j])
    idx4 = torch.tensor(diag_idx, device=device)

    # 5: anti-diagonal ↙
    anti_diag_idx = []
    for s in range(H + W - 1):
        for i in range(max(0, s - W + 1), min(H, s + 1)):
            j = W - 1 - (s - i)
            if 0 <= j < W:
                anti_diag_idx.append(mat[i, j])
    idx5 = torch.tensor(anti_diag_idx, device=device)

    # 6: reverse of idx4
    idx6 = torch.flip(idx4, [0])

    # 7: reverse of idx5
    idx7 = torch.flip(idx5, [0])

    # Pad to same length L (in case diag sequences differ due to non-square)
    def pad_to(x):
        if x.shape[0] < L:
            return torch.cat([x, torch.zeros(L - x.shape[0], dtype=torch.long, device=device)])
        elif x.shape[0] > L:
            return x[:L]
        else:
            return x

    idx4 = pad_to(idx4)
    idx5 = pad_to(idx5)
    idx6 = pad_to(idx6)
    idx7 = pad_to(idx7)

    return (idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7)


# ============================
# Triton Kernels (FIXED: no list indexing in kernel)
# ============================

@triton.jit
def triton_gather_8directions(
    x_ptr,          # (B, C, L)
    y_ptr,          # (B, 8, C, L)
    idx0_ptr, idx1_ptr, idx2_ptr, idx3_ptr,
    idx4_ptr, idx5_ptr, idx6_ptr, idx7_ptr,
    B, C, L,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_l = tl.arange(0, BLOCK_SIZE)
    x_base = x_ptr + pid_b * C * L + pid_c * L

    # k = 0
    y_base = y_ptr + pid_b * 8 * C * L + 0 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        src_idx = tl.load(idx0_ptr + offs, mask=mask)
        val = tl.load(x_base + src_idx, mask=mask)
        tl.store(y_base + offs, val, mask=mask)

    # k = 1
    y_base = y_ptr + pid_b * 8 * C * L + 1 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        src_idx = tl.load(idx1_ptr + offs, mask=mask)
        val = tl.load(x_base + src_idx, mask=mask)
        tl.store(y_base + offs, val, mask=mask)

    # k = 2
    y_base = y_ptr + pid_b * 8 * C * L + 2 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        src_idx = tl.load(idx2_ptr + offs, mask=mask)
        val = tl.load(x_base + src_idx, mask=mask)
        tl.store(y_base + offs, val, mask=mask)

    # k = 3
    y_base = y_ptr + pid_b * 8 * C * L + 3 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        src_idx = tl.load(idx3_ptr + offs, mask=mask)
        val = tl.load(x_base + src_idx, mask=mask)
        tl.store(y_base + offs, val, mask=mask)

    # k = 4
    y_base = y_ptr + pid_b * 8 * C * L + 4 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        src_idx = tl.load(idx4_ptr + offs, mask=mask)
        val = tl.load(x_base + src_idx, mask=mask)
        tl.store(y_base + offs, val, mask=mask)

    # k = 5
    y_base = y_ptr + pid_b * 8 * C * L + 5 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        src_idx = tl.load(idx5_ptr + offs, mask=mask)
        val = tl.load(x_base + src_idx, mask=mask)
        tl.store(y_base + offs, val, mask=mask)

    # k = 6
    y_base = y_ptr + pid_b * 8 * C * L + 6 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        src_idx = tl.load(idx6_ptr + offs, mask=mask)
        val = tl.load(x_base + src_idx, mask=mask)
        tl.store(y_base + offs, val, mask=mask)

    # k = 7
    y_base = y_ptr + pid_b * 8 * C * L + 7 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        src_idx = tl.load(idx7_ptr + offs, mask=mask)
        val = tl.load(x_base + src_idx, mask=mask)
        tl.store(y_base + offs, val, mask=mask)


@triton.jit
def triton_scatter_8directions(
    y_ptr,          # (B, 8, C, L)
    x_ptr,          # (B, C, L)
    idx0_ptr, idx1_ptr, idx2_ptr, idx3_ptr,
    idx4_ptr, idx5_ptr, idx6_ptr, idx7_ptr,
    B, C, L,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_l = tl.arange(0, BLOCK_SIZE)
    x_base = x_ptr + pid_b * C * L + pid_c * L

    # Zero init
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        tl.store(x_base + offs, 0.0, mask=mask)

    # k = 0
    y_base = y_ptr + pid_b * 8 * C * L + 0 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        dst_idx = tl.load(idx0_ptr + offs, mask=mask)
        val = tl.load(y_base + offs, mask=mask)
        tl.atomic_add(x_base + dst_idx, val, mask=mask)

    # k = 1
    y_base = y_ptr + pid_b * 8 * C * L + 1 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        dst_idx = tl.load(idx1_ptr + offs, mask=mask)
        val = tl.load(y_base + offs, mask=mask)
        tl.atomic_add(x_base + dst_idx, val, mask=mask)

    # k = 2
    y_base = y_ptr + pid_b * 8 * C * L + 2 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        dst_idx = tl.load(idx2_ptr + offs, mask=mask)
        val = tl.load(y_base + offs, mask=mask)
        tl.atomic_add(x_base + dst_idx, val, mask=mask)

    # k = 3
    y_base = y_ptr + pid_b * 8 * C * L + 3 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        dst_idx = tl.load(idx3_ptr + offs, mask=mask)
        val = tl.load(y_base + offs, mask=mask)
        tl.atomic_add(x_base + dst_idx, val, mask=mask)

    # k = 4
    y_base = y_ptr + pid_b * 8 * C * L + 4 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        dst_idx = tl.load(idx4_ptr + offs, mask=mask)
        val = tl.load(y_base + offs, mask=mask)
        tl.atomic_add(x_base + dst_idx, val, mask=mask)

    # k = 5
    y_base = y_ptr + pid_b * 8 * C * L + 5 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        dst_idx = tl.load(idx5_ptr + offs, mask=mask)
        val = tl.load(y_base + offs, mask=mask)
        tl.atomic_add(x_base + dst_idx, val, mask=mask)

    # k = 6
    y_base = y_ptr + pid_b * 8 * C * L + 6 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        dst_idx = tl.load(idx6_ptr + offs, mask=mask)
        val = tl.load(y_base + offs, mask=mask)
        tl.atomic_add(x_base + dst_idx, val, mask=mask)

    # k = 7
    y_base = y_ptr + pid_b * 8 * C * L + 7 * C * L + pid_c * L
    for lo in range(0, L, BLOCK_SIZE):
        offs = lo + offs_l
        mask = offs < L
        dst_idx = tl.load(idx7_ptr + offs, mask=mask)
        val = tl.load(y_base + offs, mask=mask)
        tl.atomic_add(x_base + dst_idx, val, mask=mask)


# ============================
# Autograd Functions (unchanged)
# ============================

class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        L = H * W
        ctx.shape = (B, C, H, W)
        x = x.contiguous()

        indices = _get_scan_indices(H, W, x.device.type)
        ctx.saved_indices = indices

        y = torch.empty((B, 8, C, L), device=x.device, dtype=x.dtype)

        grid = lambda meta: (B, C)
        BLOCK_SIZE = min(triton.next_power_of_2(L), 1024)
        triton_gather_8directions[grid](
            x.as_strided((B, C, L), (C*L, L, 1)),
            y,
            *indices,
            B, C, L,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        B, _, C, L = y.shape
        H, W = ctx.shape[2], ctx.shape[3]
        y = y.contiguous()
        x = torch.empty((B, C, L), device=y.device, dtype=y.dtype)

        grid = lambda meta: (B, C)
        BLOCK_SIZE = min(triton.next_power_of_2(L), 1024)
        triton_scatter_8directions[grid](
            y,
            x,
            *ctx.saved_indices,
            B, C, L,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return x.view(B, C, H, W)


class CrossMergeTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor):
        B, K, C, H, W = y.shape
        assert K == 8, f"Expected 8 directions, got {K}"
        L = H * W
        ctx.shape = (B, C, H, W)
        y = y.contiguous().view(B, 8, C, L)

        indices = _get_scan_indices(H, W, y.device.type)
        ctx.saved_indices = indices

        x = torch.empty((B, C, L), device=y.device, dtype=y.dtype)

        grid = lambda meta: (B, C)
        BLOCK_SIZE = min(triton.next_power_of_2(L), 1024)
        triton_scatter_8directions[grid](
            y,
            x,
            *indices,
            B, C, L,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return x

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, L = x.shape
        H, W = ctx.shape[2], ctx.shape[3]
        x = x.contiguous()
        y = torch.empty((B, 8, C, L), device=x.device, dtype=x.dtype)

        grid = lambda meta: (B, C)
        BLOCK_SIZE = min(triton.next_power_of_2(L), 1024)
        triton_gather_8directions[grid](
            x.as_strided((B, C, L), (C*L, L, 1)),
            y,
            *ctx.saved_indices,
            B, C, L,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y.view(B, 8, C, H, W)


# ============================
# Backward Compatibility Wrappers
# ============================

class CrossScanTriton4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y8 = CrossScanTriton.apply(x)
        y4 = y8[:, :4]
        ctx.save_for_backward(y8)
        return y4

    @staticmethod
    def backward(ctx, y4):
        y8, = ctx.saved_tensors
        y8_grad = torch.zeros_like(y8)
        y8_grad[:, :4] = y4
        x_grad = CrossScanTriton.backward(ctx, y8_grad)
        return x_grad


# Optional: 2-direction UV (for lightweight models) — kept as fallback
class CrossScanTriton1b1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, K, C, H, W = x.shape
        assert K == 8
        L = H * W
        ctx.shape = (B, C, H, W)
        x = x.contiguous().view(B, 8, C, L)
        indices = _get_scan_indices(H, W, x.device.type)
        ctx.saved_indices = indices

        y = torch.empty_like(x)
        for k in range(8):
            idx = indices[k][:L]
            y[:, k] = x[:, k].gather(-1, idx.expand(B, C, -1))
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        B, _, C, L = y.shape
        H, W = ctx.shape[2], ctx.shape[3]
        indices = ctx.saved_indices
        x = torch.zeros_like(y)
        for k in range(8):
            idx = indices[k][:L]
            x[:, k].scatter_add_(-1, idx.expand(B, C, -1), y[:, k])
        return x.view(B, 8, C, H, W)