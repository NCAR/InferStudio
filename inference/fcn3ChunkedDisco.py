"""
Lower FourCastNet3's peak GPU memory by doing torch_harmonics' DISCO
convolution weight contraction in latitude chunks.

DiscreteContinuousConvS2.forward contracts its (B, C, K, H, W) intermediate
with the weights in one einsum. torch makes a permuted copy of that whole
intermediate for the einsum - about 20 GiB at FCN3's 0.25 degree grid - which
runs FCN3 out of memory on a 40 GB A100. The contraction is independent per
latitude row, so doing it on slices of rows gives the same result while each
temporary copy is only a slice's size.

    import fcn3ChunkedDisco
    fcn3ChunkedDisco.apply(n_chunks=8)   # before running the model
    fcn3ChunkedDisco.remove()            # restore the original forward
"""

import torch
from torch_harmonics.disco import convolution as _conv

_original_forward = _conv.DiscreteContinuousConvS2.forward
N_CHUNKS = 8


def _chunked_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Same as DiscreteContinuousConvS2.forward (torch_harmonics 0.9.1) except
    # the weight einsum, which runs over N_CHUNKS slices of latitude rows.
    if self.optimized_kernel:
        x = _conv._disco_s2_contraction_optimized(
            x, self.psi_roff_idx, self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx,
            self.psi_vals, self.kernel_size, self.nlat_out, self.nlon_out
        )
    else:
        x = _conv._disco_s2_contraction_torch(x, self.psi.to(x.device), self.nlon_out)

    B, C, K, H, W = x.shape
    x = x.reshape(B, self.groups, self.groupsize, K, H, W)
    weight = self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2])

    out = None
    step = -(-H // N_CHUNKS)
    for h0 in range(0, H, step):
        part = torch.einsum("bgckxy,gock->bgoxy", x[..., h0:h0 + step, :], weight)
        if out is None:
            out = part.new_empty(part.shape[:-2] + (H, W))
        out[..., h0:h0 + step, :] = part
        del part
    out = out.reshape(B, -1, H, W)

    if self.bias is not None:
        out = out + self.bias.reshape(1, -1, 1, 1)

    return out


def apply(n_chunks: int = 8) -> None:
    global N_CHUNKS
    N_CHUNKS = n_chunks
    _conv.DiscreteContinuousConvS2.forward = _chunked_forward


def remove() -> None:
    _conv.DiscreteContinuousConvS2.forward = _original_forward
