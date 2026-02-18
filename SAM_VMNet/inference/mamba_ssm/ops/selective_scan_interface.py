"""
Pure PyTorch implementation of Mamba's selective scan.
Replaces the CUDA kernel for CPU/MPS inference.

This is a direct reimplementation of selective_scan_fn from
mamba_ssm.ops.selective_scan_interface using only standard PyTorch ops.
Slow (sequential scan) but correct and runs on any device.
"""

import torch
import torch.nn.functional as F


def selective_scan_fn(u, delta, A, B, C, D=None, z=None,
                      delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    Selective scan (S6) — the core Mamba operation.

    Args:
        u:     (B, dim, L)     input sequence
        delta: (B, dim, L)     timestep / gate
        A:     (dim, N)        state transition (negative values)
        B:     (B, G, N, L)    input-to-state matrix (G groups)
        C:     (B, G, N, L)    state-to-output matrix (G groups)
        D:     (dim,)          skip connection
        z:     (B, dim, L)     optional SiLU gate
        delta_bias: (dim,)     bias added to delta
        delta_softplus: bool   apply softplus to delta
        return_last_state: bool

    Returns:
        y: (B, dim, L)  or  (y, last_state) if return_last_state
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1).float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, L = u.shape
    N = A.shape[1]
    G = B.shape[1]

    A = A.float()
    B = B.float()
    C = C.float()

    # Expand groups: (B, G, N, L) -> (B, dim, N, L)
    if G < dim:
        expand = dim // G
        B = B.unsqueeze(2).expand(-1, G, expand, N, L).reshape(batch, dim, N, L)
        C = C.unsqueeze(2).expand(-1, G, expand, N, L).reshape(batch, dim, N, L)

    # Hidden state
    h = torch.zeros(batch, dim, N, device=u.device, dtype=torch.float32)
    ys = []

    for l in range(L):
        dt = delta[:, :, l]                                   # (B, dim)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))     # (B, dim, N)
        dB = dt.unsqueeze(-1) * B[:, :, :, l]                 # (B, dim, N)
        h = dA * h + dB * u[:, :, l].unsqueeze(-1)            # (B, dim, N)
        y_l = (C[:, :, :, l] * h).sum(-1)                     # (B, dim)
        ys.append(y_l)

    y = torch.stack(ys, dim=-1)  # (B, dim, L)

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1).float()

    if z is not None:
        y = y * F.silu(z.float())

    y = y.to(dtype_in)

    if return_last_state:
        return y, h
    return y


# Alias expected by vmamba.py
selective_scan_ref = selective_scan_fn
