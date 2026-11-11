from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_AVAILABLE = False
try:
    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")
        _CUDA_AVAILABLE = True
except Exception:
    pass

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception as _e:
    import sys
    print(f"[mamba_blocks] WARNING: mamba_ssm import failed: {_e!r}", file=sys.stderr)
    print(f"[mamba_blocks]   _CUDA_AVAILABLE before import: {_CUDA_AVAILABLE}", file=sys.stderr)
    Mamba = None
    _HAS_MAMBA = False


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class TokenMixer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        if _HAS_MAMBA:
            self.mixer = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.mixer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class DeformableSampler3D(nn.Module):
    def __init__(self, channels: int, offset_limit: float = 2.0):
        super().__init__()
        self.offset_limit = float(offset_limit)
        self.offset_head = nn.Conv3d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.register_buffer("last_offset_l1", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_offset_tv", torch.tensor(0.0), persistent=False)

    @staticmethod
    def _base_grid(d: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        z = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)

    @staticmethod
    def _offset_tv(off_vox: torch.Tensor) -> torch.Tensor:
        tv = off_vox.new_tensor(0.0)
        if off_vox.shape[2] > 1:
            tv = tv + (off_vox[:, :, 1:, :, :] - off_vox[:, :, :-1, :, :]).abs().mean()
        if off_vox.shape[3] > 1:
            tv = tv + (off_vox[:, :, :, 1:, :] - off_vox[:, :, :, :-1, :]).abs().mean()
        if off_vox.shape[4] > 1:
            tv = tv + (off_vox[:, :, :, :, 1:] - off_vox[:, :, :, :, :-1]).abs().mean()
        return tv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        off_vox = torch.tanh(self.offset_head(x)) * self.offset_limit
        self.last_offset_l1 = off_vox.abs().mean().detach()
        self.last_offset_tv = self._offset_tv(off_vox).detach()

        sx = 2.0 / max(w - 1, 1)
        sy = 2.0 / max(h - 1, 1)
        sz = 2.0 / max(d - 1, 1)

        off_norm = off_vox.clone()
        off_norm[:, 0] = off_norm[:, 0] * sx
        off_norm[:, 1] = off_norm[:, 1] * sy
        off_norm[:, 2] = off_norm[:, 2] * sz
        off_norm = off_norm.permute(0, 2, 3, 4, 1).contiguous()

        base = self._base_grid(d, h, w, x.device, x.dtype).expand(b, -1, -1, -1, -1)
        grid = base + off_norm
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class SpatialFrequencyMambaBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.deformable_sampler = DeformableSampler3D(channels, offset_limit=2.0)
        self.deformable_sampler_freq = DeformableSampler3D(channels, offset_limit=1.5)
        self.norm = nn.LayerNorm(channels)
        self.spatial_mixer = TokenMixer(channels)
        self.freq_mixer = TokenMixer(channels)
        self.branch_gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_norm = nn.InstanceNorm3d(channels, affine=True)

    def _to_tokens(self, x: torch.Tensor):
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return tokens, (d, h, w)

    def _to_volume(self, tokens: torch.Tensor, shape):
        d, h, w = shape
        b, n, c = tokens.shape
        assert n == d * h * w, "Token count does not match volume shape."
        return tokens.transpose(1, 2).contiguous().view(b, c, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_spatial = self.deformable_sampler(x)
        x_freq = self.deformable_sampler_freq(x)

        tokens, shape = self._to_tokens(x_spatial)
        tokens = self.norm(tokens)
        spatial_tokens = self.spatial_mixer(tokens)

        freq_tokens_in, freq_shape = self._to_tokens(x_freq)
        freq_tokens_in = self.norm(freq_tokens_in)
        freq_volume = self._to_volume(freq_tokens_in, freq_shape)
        freq_complex = torch.fft.fftn(freq_volume, dim=(2, 3, 4), norm="ortho")
        freq_mag = torch.log1p(torch.abs(freq_complex))
        freq_tokens, _ = self._to_tokens(freq_mag)
        freq_tokens = self.freq_mixer(freq_tokens)

        fusion_in = torch.cat([spatial_tokens, freq_tokens], dim=-1)
        gate = self.branch_gate(fusion_in)
        fused = self.fuse(torch.cat([spatial_tokens * gate, freq_tokens * (1.0 - gate)], dim=-1))
        fused_volume = self._to_volume(fused, shape)
        out = self.out_proj(fused_volume)
        out = self.out_norm(out)
        return F.leaky_relu(out + residual, negative_slope=0.01, inplace=True)


class SimplifiedMambaBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mixer = TokenMixer(channels)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.out_norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.norm(tokens)
        tokens = self.mixer(tokens)
        out = tokens.transpose(1, 2).contiguous().view(b, c, d, h, w)
        out = self.proj(out)
        out = self.out_norm(out)
        return F.leaky_relu(out + residual, negative_slope=0.01, inplace=True)
