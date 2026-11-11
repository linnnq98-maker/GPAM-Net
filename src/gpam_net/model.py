"""Boundary-aware SP models without ring auxiliary supervision for the joint cascade route."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_blocks import (
    ConvNormAct,
    ResidualConvBlock,
    SimplifiedMambaBlock,
    SpatialFrequencyMambaBlock,
)


def _make_mamba_block(channels: int, block_type: str) -> nn.Module:
    if block_type == "simplified":
        return SimplifiedMambaBlock(channels)
    if block_type == "spatial_freq":
        return SpatialFrequencyMambaBlock(channels)
    raise ValueError(f"Unknown mamba_block={block_type}")


class DinoV3SliceAdapter(nn.Module):
    """Frozen DINOv3 2.5D feature branch for 3D MRI patches."""

    def __init__(
        self,
        repo_dir: str,
        weights: str,
        model_name: str = "dinov3_vits16",
        image_size: int = 224,
        out_channels: int = 8,
    ):
        super().__init__()
        if not repo_dir:
            raise ValueError("DINOv3 enabled but repo_dir is empty.")
        if not weights:
            raise ValueError("DINOv3 enabled but weights is empty.")
        repo_path = Path(repo_dir)
        if not repo_path.exists():
            raise FileNotFoundError(f"DINOv3 repo not found: {repo_path}")
        weights_path = Path(weights)
        weights_arg = str(weights_path) if weights_path.exists() else weights

        self.image_size = int(image_size)
        self.out_channels = int(out_channels)
        self.backbone = torch.hub.load(
            str(repo_path),
            model_name,
            source="local",
            weights=weights_arg,
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.proj = nn.LazyConv2d(self.out_channels, kernel_size=1)

    @staticmethod
    def _to_unit(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -3.0, 3.0)
        return (x + 3.0) / 6.0

    def _make_rgb(self, inv2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        d = inv2.shape[2]
        z = d // 2
        inv2_2d = self._to_unit(inv2[:, 0, z])
        flair_2d = self._to_unit(flair[:, 0, z])
        mean_2d = 0.5 * (inv2_2d + flair_2d)
        img = torch.stack([inv2_2d, flair_2d, mean_2d], dim=1)
        img = F.interpolate(img, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mean = img.new_tensor([0.430, 0.411, 0.296]).view(1, 3, 1, 1)
        std = img.new_tensor([0.213, 0.156, 0.143]).view(1, 3, 1, 1)
        return (img - mean) / std

    @staticmethod
    def _extract_feature_map(dino_out) -> torch.Tensor:
        if isinstance(dino_out, dict):
            for key in ("x_norm_patchtokens", "x_prenorm", "patch_tokens"):
                if key in dino_out:
                    dino_out = dino_out[key]
                    break
            else:
                dino_out = next(v for v in dino_out.values() if torch.is_tensor(v))
        if isinstance(dino_out, (list, tuple)):
            dino_out = dino_out[-1]
        if dino_out.ndim == 4:
            return dino_out
        if dino_out.ndim == 3:
            b, n, c = dino_out.shape
            side = int(math.sqrt(n))
            if side * side != n:
                side = int(math.sqrt(max(1, n - 1)))
                dino_out = dino_out[:, : side * side]
            return dino_out.transpose(1, 2).reshape(b, c, side, side)
        if dino_out.ndim == 2:
            return dino_out[:, :, None, None]
        raise ValueError(f"Unsupported DINOv3 output shape: {tuple(dino_out.shape)}")

    def forward(self, inv2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = inv2.shape
        img = self._make_rgb(inv2, flair)
        with torch.no_grad():
            if hasattr(self.backbone, "forward_features"):
                out = self.backbone.forward_features(img)
            else:
                out = self.backbone(img)
            fmap = self._extract_feature_map(out).detach()
        fmap = self.proj(fmap)
        fmap = F.interpolate(fmap, size=(h, w), mode="bilinear", align_corners=False)
        return fmap[:, :, None].expand(b, self.out_channels, d, h, w)


class SPLogitAdapter(nn.Module):
    """Lightweight SP adapter that adds a bounded correction to segmentation logits."""

    def __init__(self, sp_channels: int, hidden_channels: int = 16):
        super().__init__()
        hidden = int(hidden_channels)
        if sp_channels <= 0:
            raise ValueError("SPLogitAdapter requires sp_channels > 0")
        if hidden < 4:
            raise ValueError("sp_adapter_channels must be >= 4")
        mid = max(4, hidden // 2)

        self.net = nn.Sequential(
            nn.Conv3d(sp_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(hidden, 8), hidden),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(mid, 8), mid),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, sp: torch.Tensor, out_shape: tuple[int, int, int]) -> torch.Tensor:
        if sp.shape[2:] != out_shape:
            sp = F.interpolate(sp, size=out_shape, mode="trilinear", align_corners=False)
        return torch.tanh(self.net(sp))


class _Encoder(nn.Module):
    def __init__(self, in_ch: int, channels: tuple[int, ...], block_type: str):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.stem = ConvNormAct(in_ch, c0)
        self.enc0 = ResidualConvBlock(c0)
        self.down1 = ConvNormAct(c0, c1, k=3, s=2, p=1)
        self.enc1 = ResidualConvBlock(c1)
        self.down2 = ConvNormAct(c1, c2, k=3, s=2, p=1)
        self.enc2 = ResidualConvBlock(c2)
        self.down3 = ConvNormAct(c2, c3, k=3, s=2, p=1)
        self.enc3 = _make_mamba_block(c3, block_type)
        self.down4 = ConvNormAct(c3, c4, k=3, s=2, p=1)
        self.enc4 = _make_mamba_block(c4, block_type)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.enc0(self.stem(x))
        x1 = self.enc1(self.down1(x0))
        x2 = self.enc2(self.down2(x1))
        x3 = self.enc3(self.down3(x2))
        x4 = self.enc4(self.down4(x3))
        return [x0, x1, x2, x3, x4]


class _AttentionGate(nn.Module):
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.gate_proj = nn.Conv3d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.skip_proj = nn.Conv3d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="trilinear", align_corners=False)
        attn = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * attn


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_gate: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = _AttentionGate(out_ch, skip_ch, max(out_ch // 2, 8)) if attention_gate else None
        self.fuse = nn.Sequential(
            ConvNormAct(out_ch + skip_ch, out_ch),
            ResidualConvBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        if self.attn is not None:
            skip = self.attn(x, skip)
        return self.fuse(torch.cat([x, skip], dim=1))


class _BoundaryAwareGatedFusionBlock(nn.Module):
    """Gated INV2/FLAIR fusion modulated by an INV2 structural-boundary proxy."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels
        self.net = nn.Sequential(
            nn.Conv3d(channels * 4 + 3, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1 if hidden < 8 else min(8, hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.net[-1].bias, 1.0)

    def forward(
        self,
        f_inv2: torch.Tensor,
        f_flair: torch.Tensor,
        m_inv2: torch.Tensor,
        m_flair: torch.Tensor,
        boundary: torch.Tensor | None,
    ) -> torch.Tensor:
        mi = m_inv2.view(-1, 1, 1, 1, 1).to(dtype=f_inv2.dtype, device=f_inv2.device)
        mf = m_flair.view(-1, 1, 1, 1, 1).to(dtype=f_inv2.dtype, device=f_inv2.device)
        mi_map = mi.expand(-1, 1, *f_inv2.shape[2:])
        mf_map = mf.expand(-1, 1, *f_inv2.shape[2:])
        if boundary is None:
            boundary_map = torch.zeros_like(mi_map)
        else:
            boundary_map = boundary.to(dtype=f_inv2.dtype, device=f_inv2.device)
            if boundary_map.shape[2:] != f_inv2.shape[2:]:
                boundary_map = F.interpolate(boundary_map, size=f_inv2.shape[2:], mode="trilinear", align_corners=False)
        gate_in = torch.cat(
            [f_inv2, f_flair, f_inv2 - f_flair, f_inv2 * f_flair, boundary_map, mi_map, mf_map],
            dim=1,
        )
        gate = torch.sigmoid(self.net(gate_in))
        return gate * f_inv2 + (1.0 - gate) * f_flair


class BoundaryUncertaintyMambaUNet(nn.Module):
    """Dual-encoder INV2+FLAIR+SP model with boundary-aware fusion and uncertainty head."""

    def __init__(
        self,
        in_channels: int = 2,
        channels: tuple[int, ...] = (32, 64, 128, 256, 320),
        deep_supervision: bool = True,
        mamba_block: str = "simplified",
        attention_gates: bool = True,
        boundary_blur_kernel: int = 3,
        sp_channels: int = 0,
        sp_dropout: float = 0.0,
    ):
        super().__init__()
        if in_channels != 2:
            raise ValueError("BoundaryUncertaintyMambaUNet expects 2 input channels (INV2 + FLAIR).")
        c0, c1, c2, c3, c4 = channels
        self.deep_supervision = deep_supervision
        self.sp_channels = int(sp_channels)
        self.sp_dropout = float(sp_dropout)
        if self.sp_channels < 0:
            raise ValueError("sp_channels must be >= 0")
        self.boundary_blur_kernel = max(1, int(boundary_blur_kernel))
        if self.boundary_blur_kernel % 2 == 0:
            self.boundary_blur_kernel += 1

        branch_in_channels = 1 + self.sp_channels
        self.enc_inv2 = _Encoder(branch_in_channels, channels, mamba_block)
        self.enc_flair = _Encoder(branch_in_channels, channels, mamba_block)
        self.fusions = nn.ModuleList([_BoundaryAwareGatedFusionBlock(c) for c in channels])

        self.bottleneck = _make_mamba_block(c4, mamba_block)

        self.up4 = _UpBlock(c4, c3, c3, attention_gate=attention_gates)
        self.up3 = _UpBlock(c3, c2, c2, attention_gate=attention_gates)
        self.up2 = _UpBlock(c2, c1, c1, attention_gate=attention_gates)
        self.up1 = _UpBlock(c1, c0, c0, attention_gate=attention_gates)

        self.seg_head = nn.Conv3d(c0, 1, kernel_size=1)
        self.uncertainty_head = nn.Sequential(
            ConvNormAct(c0, c0, k=3, s=1, p=1),
            nn.Conv3d(c0, 1, kernel_size=1),
        )
        if deep_supervision:
            self.seg_ds1 = nn.Conv3d(c1, 1, kernel_size=1)
            self.seg_ds2 = nn.Conv3d(c2, 1, kernel_size=1)
            self.seg_ds3 = nn.Conv3d(c3, 1, kernel_size=1)

    def _structure_boundary(self, inv2: torch.Tensor) -> torch.Tensor:
        dz = F.pad(inv2[:, :, 1:, :, :] - inv2[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
        dy = F.pad(inv2[:, :, :, 1:, :] - inv2[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
        dx = F.pad(inv2[:, :, :, :, 1:] - inv2[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        mag = torch.sqrt(dx.square() + dy.square() + dz.square() + 1e-6)
        mean = mag.mean(dim=(2, 3, 4), keepdim=True)
        std = mag.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        boundary = torch.sigmoid((mag - mean) / std)
        if self.boundary_blur_kernel > 1:
            pad = self.boundary_blur_kernel // 2
            boundary = F.max_pool3d(boundary, kernel_size=self.boundary_blur_kernel, stride=1, padding=pad)
        return boundary.clamp(0.0, 1.0)

    def forward(
        self,
        inv2: torch.Tensor,
        flair: torch.Tensor | None = None,
        m_inv2: torch.Tensor | None = None,
        m_flair: torch.Tensor | None = None,
        sp: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if flair is None:
            if inv2.ndim != 5 or inv2.shape[1] < 2:
                raise ValueError("Expected either [B,2(+SP),D,H,W] or separate INV2/FLAIR tensors.")
            if sp is None and inv2.shape[1] > 2:
                sp = inv2[:, 2:]
            flair = inv2[:, 1:2]
            inv2 = inv2[:, :1]
        elif inv2.ndim != 5 or flair.ndim != 5 or inv2.shape[1] != 1 or flair.shape[1] != 1:
            raise ValueError("Expected separate tensors shaped [B,1,D,H,W].")

        if self.sp_channels > 0:
            if sp is None:
                raise ValueError(
                    f"This model was created with sp_channels={self.sp_channels}; "
                    "pass SP maps via sp=... or concatenate them after INV2/FLAIR."
                )
            if sp.ndim != 5 or sp.shape[1] < self.sp_channels:
                raise ValueError(f"Expected SP tensor [B,{self.sp_channels},D,H,W], got {tuple(sp.shape)}")
            sp = sp[:, :self.sp_channels].to(dtype=inv2.dtype, device=inv2.device)
            if sp.shape[2:] != inv2.shape[2:]:
                sp = F.interpolate(sp, size=inv2.shape[2:], mode="trilinear", align_corners=False)
            if self.training and self.sp_dropout > 0:
                keep = (torch.rand(sp.shape[0], 1, 1, 1, 1, device=sp.device) >= self.sp_dropout).to(sp.dtype)
                sp = sp * keep
        elif sp is not None:
            sp = None

        if m_inv2 is None:
            m_inv2 = torch.ones(inv2.shape[0], device=inv2.device, dtype=inv2.dtype)
        if m_flair is None:
            m_flair = torch.ones(inv2.shape[0], device=inv2.device, dtype=inv2.dtype)

        mf_b = m_flair.view(-1, 1, 1, 1, 1).to(dtype=inv2.dtype, device=inv2.device)
        flair_in = mf_b * flair + (1.0 - mf_b) * inv2
        boundary = self._structure_boundary(inv2)
        inv2_in = torch.cat([inv2, sp], dim=1) if sp is not None else inv2
        flair_in = torch.cat([flair_in, sp], dim=1) if sp is not None else flair_in

        f_inv2 = self.enc_inv2(inv2_in)
        f_flair = self.enc_flair(flair_in)
        fused = [
            fusion(a, b, m_inv2, m_flair, boundary)
            for fusion, a, b in zip(self.fusions, f_inv2, f_flair)
        ]

        xb = self.bottleneck(fused[4])
        d3 = self.up4(xb, fused[3])
        d2 = self.up3(d3, fused[2])
        d1 = self.up2(d2, fused[1])
        d0 = self.up1(d1, fused[0])

        seg = self.seg_head(d0)
        uncertainty_logit = self.uncertainty_head(d0)
        uncertainty = torch.sigmoid(uncertainty_logit)
        # Compatibility only: this branch intentionally has no ring auxiliary head/loss.
        ring = torch.zeros_like(seg)
        out: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "seg": seg,
            "ring": ring,
            "heat": ring,
            "uncertainty": uncertainty,
            "uncertainty_logit": uncertainty_logit,
            "structure_boundary": boundary,
        }
        if self.deep_supervision and self.training:
            out["seg_ds"] = [seg, self.seg_ds1(d1), self.seg_ds2(d2), self.seg_ds3(d3)]
            out["features_ds"] = [d0, d1, d2, d3]
        return out


class JointCascadeDINOStage1(nn.Module):
    """Stage-1 lesion+uncertainty model with optional SP and frozen DINOv3 residual adapter."""

    def __init__(
        self,
        in_channels: int = 2,
        deep_supervision: bool = True,
        mamba_block: str = "simplified",
        attention_gates: bool = True,
        dinov3_repo: str | None = None,
        dinov3_weights: str | None = None,
        dinov3_model: str = "dinov3_vits16",
        dinov3_image_size: int = 224,
        dinov3_aux_channels: int = 8,
        boundary_blur_kernel: int = 3,
        sp_channels: int = 0,
        sp_dropout: float = 0.0,
        sp_fusion_mode: str = "early",
        sp_adapter_channels: int = 16,
        sp_gate_max: float = 0.15,
        sp_warmup_epochs: int = 0,
        sp_rampup_epochs: int = 1,
    ):
        super().__init__()
        sp_fusion_mode = str(sp_fusion_mode).lower().strip()
        if sp_fusion_mode not in {"early", "logit_adapter"}:
            raise ValueError(f"Unsupported sp_fusion_mode={sp_fusion_mode!r}")
        self.sp_channels = int(sp_channels)
        self.sp_dropout = float(sp_dropout)
        self.sp_fusion_mode = sp_fusion_mode
        self.sp_gate_max = float(sp_gate_max)
        self.sp_warmup_epochs = int(sp_warmup_epochs)
        self.sp_rampup_epochs = max(1, int(sp_rampup_epochs))
        base_sp_channels = self.sp_channels if self.sp_fusion_mode == "early" else 0
        self.base = BoundaryUncertaintyMambaUNet(
            in_channels=in_channels,
            deep_supervision=deep_supervision,
            mamba_block=mamba_block,
            attention_gates=attention_gates,
            boundary_blur_kernel=boundary_blur_kernel,
            sp_channels=base_sp_channels,
            sp_dropout=sp_dropout if self.sp_fusion_mode == "early" else 0.0,
        )
        self.sp_adapter = None
        if self.sp_channels > 0 and self.sp_fusion_mode == "logit_adapter":
            self.sp_adapter = SPLogitAdapter(self.sp_channels, hidden_channels=sp_adapter_channels)
        self.dinov3_enabled = bool(dinov3_repo and dinov3_weights)
        self.dino = None
        self.aux_head = None
        if self.dinov3_enabled:
            self.dino = DinoV3SliceAdapter(
                repo_dir=dinov3_repo or "",
                weights=dinov3_weights or "",
                model_name=dinov3_model,
                image_size=dinov3_image_size,
                out_channels=dinov3_aux_channels,
            )
            self.aux_head = nn.Sequential(
                nn.Conv3d(1 + dinov3_aux_channels, 16, kernel_size=3, padding=1),
                nn.InstanceNorm3d(16, affine=True),
                nn.SiLU(inplace=True),
                nn.Conv3d(16, 1, kernel_size=1),
            )
            nn.init.zeros_(self.aux_head[-1].weight)
            nn.init.zeros_(self.aux_head[-1].bias)

    def _sp_alpha(self, epoch: int | None = None, sp_alpha: float | None = None) -> float:
        if self.sp_channels <= 0:
            return 1.0
        if sp_alpha is not None:
            return float(sp_alpha)
        if epoch is None:
            return 1.0
        if epoch < self.sp_warmup_epochs:
            return 0.0
        return min(1.0, float(epoch - self.sp_warmup_epochs) / float(self.sp_rampup_epochs))

    def forward(
        self,
        inv2: torch.Tensor,
        flair: torch.Tensor | None = None,
        m_inv2: torch.Tensor | None = None,
        m_flair: torch.Tensor | None = None,
        sp: torch.Tensor | None = None,
        epoch: int | None = None,
        sp_alpha: float | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if flair is None:
            if inv2.ndim != 5 or inv2.shape[1] < 2:
                raise ValueError("Expected [B,2(+SP),D,H,W] or separate INV2/FLAIR tensors.")
            if sp is None and inv2.shape[1] > 2:
                sp = inv2[:, 2:]
            flair = inv2[:, 1:2]
            inv2_in = inv2[:, :1]
        else:
            inv2_in = inv2

        base_sp = None
        if self.sp_fusion_mode == "early":
            if sp is None and self.sp_channels > 0:
                raise ValueError("sp_fusion_mode='early' requires SP maps")
            if sp is not None:
                alpha = self._sp_alpha(epoch=epoch, sp_alpha=sp_alpha)
                base_sp = sp[:, :self.sp_channels].to(dtype=inv2_in.dtype, device=inv2_in.device)
                base_sp = base_sp * float(alpha)

        out = self.base(inv2_in, flair, m_inv2, m_flair, sp=base_sp)
        if self.sp_fusion_mode == "early" and self.sp_channels > 0:
            out = dict(out)
            out["sp_alpha"] = out["seg"].new_tensor(float(self._sp_alpha(epoch=epoch, sp_alpha=sp_alpha)))
        if self.sp_adapter is not None:
            if sp is None:
                raise ValueError("sp_fusion_mode='logit_adapter' requires SP maps")
            sp_in = sp[:, :self.sp_channels].to(dtype=out["seg"].dtype, device=out["seg"].device)
            if self.training and self.sp_dropout > 0:
                mask = torch.bernoulli(
                    torch.full(
                        (sp_in.shape[0], sp_in.shape[1], 1, 1, 1),
                        1.0 - self.sp_dropout,
                        device=sp_in.device,
                        dtype=sp_in.dtype,
                    )
                )
                sp_in = sp_in * mask / max(1.0 - self.sp_dropout, 1e-6)
            alpha = self._sp_alpha(epoch=epoch, sp_alpha=sp_alpha)
            correction = self.sp_adapter(sp_in, out["seg"].shape[2:])
            seg = out["seg"] + float(alpha) * self.sp_gate_max * correction
            out = dict(out)
            out["seg_main"] = out["seg"]
            out["sp_correction"] = correction
            out["sp_alpha"] = out["seg"].new_tensor(float(alpha))
            out["seg"] = seg
            if "seg_ds" in out:
                seg_ds = [seg]
                for i in range(1, len(out["seg_ds"])):
                    ds_shape = out["seg_ds"][i].shape[2:]
                    ds_correction = F.interpolate(correction, size=ds_shape, mode="trilinear", align_corners=False)
                    seg_ds.append(out["seg_ds"][i] + float(alpha) * self.sp_gate_max * ds_correction)
                out["seg_ds"] = seg_ds
        if self.dinov3_enabled:
            assert self.dino is not None and self.aux_head is not None
            dino_feat = self.dino(inv2_in, flair)
            corr = self.aux_head(torch.cat([out["seg"], dino_feat], dim=1))
            seg = out["seg"] + corr
            out = dict(out)
            out["seg"] = seg
            out["ring"] = torch.zeros_like(seg)
            out["heat"] = out["ring"]
            if "seg_ds" in out:
                seg_ds = list(out["seg_ds"])
                seg_ds[0] = seg
                out["seg_ds"] = seg_ds
        return out


