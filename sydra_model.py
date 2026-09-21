import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ============================================================== #
# Multi-Scale Geometric Feature Distillation (MGFD).
# ============================================================== #
class MGFD(nn.Module):
    def __init__(self, in_ch, out_ch, scales=(1, 2, 4), reduction=4):
        super().__init__()
        self.scales = scales
        hidden = max(8, out_ch // reduction)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=s, dilation=s, bias=False),
                nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
                nn.ReLU(inplace=False)
            ) for s in scales
        ])
        self.merge = nn.Sequential(
            nn.Conv2d(hidden * len(scales), out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        # Materialize a contiguous tensor before the following convolution.
        x = x.contiguous()

        feats = [conv(x) for conv in self.convs]
        out = torch.cat(feats, dim=1)
        return self.merge(out)


# ============================================================== #
# Manifold-Constrained Contextual Calibration (MCCC).
# ============================================================== #
class LowRankProjector(nn.Module):
    def __init__(self, dim, rank=16):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        return self.V(self.U(x))


class MCCC(nn.Module):
    def __init__(self, channels, reduction=16, rank=16):
        super().__init__()
        # Global context aggregation.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Channel compression and expansion.
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

        # Lightweight low-rank feature fusion.
        self.lowrank = LowRankProjector(channels, rank)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x)  # [B, C, 1, 1]
        y_flat = y.view(b, c)
        y_lowrank = self.lowrank(y_flat).view(b, c, 1, 1)

        attn = self.fc1(y_lowrank)
        attn = F.relu(attn, inplace=False)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)

        return x * attn


# ============================================================== #
# Dynamic Interference-Suppression Attention (DISA).
# ============================================================== #
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class DISA(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, dropout=0.1):
        super().__init__()
        assert dim >= 1 and dim % num_heads == 0 and window_size >= 1
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.ws_sq = window_size ** 2
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, groups=num_heads, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.mask_generator = nn.Sequential(
            nn.Conv2d(dim, max(dim // 4, 1), 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(dim // 4, 1), num_heads * self.ws_sq, 1)
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, num_heads, self.ws_sq, self.head_dim))
        self.residual_in = ResidualBlock(dim)
        self.residual_out = ResidualBlock(dim)
        self.cross_window = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        x_in = x.clone()
        x = self.residual_in(x_in)
        B, C, H, W = x.shape

        H_pad = (self.window_size - H % self.window_size) % self.window_size
        W_pad = (self.window_size - W % self.window_size) % self.window_size
        if H_pad or W_pad:
            x = F.pad(x, (0, W_pad, 0, H_pad))

        H_padded, W_padded = x.shape[2], x.shape[3]
        window_h, window_w = H_padded // self.window_size, W_padded // self.window_size
        ws_sq = self.ws_sq

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h d) (wh ws_h) (ww ws_w) -> b h (wh ww) (ws_h ws_w) d',
                      h=self.num_heads, d=self.head_dim, wh=window_h, ww=window_w,
                      ws_h=self.window_size, ws_w=self.window_size)

        actual_ws_sq = q.shape[3]
        pos_encoding = self.pos_encoding
        if actual_ws_sq != self.ws_sq:
            pos_encoding = F.interpolate(pos_encoding.permute(0, 1, 3, 2), size=actual_ws_sq,
                                         mode='linear', align_corners=False).permute(0, 1, 3, 2).contiguous()

        q = q + pos_encoding.unsqueeze(2)

        k = rearrange(k, 'b (h d) (wh ws_h) (ww ws_w) -> b h (wh ww) d (ws_h ws_w)',
                      h=self.num_heads, d=self.head_dim, wh=window_h, ww=window_w,
                      ws_h=self.window_size, ws_w=self.window_size)
        v = rearrange(v, 'b (h d) (wh ws_h) (ww ws_w) -> b h (wh ww) (ws_h ws_w) d',
                      h=self.num_heads, d=self.head_dim, wh=window_h, ww=window_w,
                      ws_h=self.window_size, ws_w=self.window_size)

        window_feat = F.avg_pool2d(x, kernel_size=self.window_size, stride=self.window_size)
        mask = self.mask_generator(window_feat)
        mask = rearrange(mask, 'b (h ws) wh ww -> b h (wh ww) ws',
                         h=self.num_heads, ws=ws_sq, wh=window_h, ww=window_w)

        attn = (q @ k) * self.scale
        attn = attn + mask.unsqueeze(-2)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, 'b h (wh ww) (ws_h ws_w) d -> b (h d) (wh ws_h) (ww ws_w)',
                        h=self.num_heads, d=self.head_dim, wh=window_h, ww=window_w,
                        ws_h=self.window_size, ws_w=self.window_size)

        if H_pad or W_pad:
            out = out[:, :, :H, :W]

        out = self.proj(out)
        out = self.cross_window(out)
        out = out + x_in
        out = self.residual_out(out)
        return out


# ============================================================== #
# SyDRA reconstructive encoder: MGFD before the bottleneck, then MCCC.
# ============================================================== #

class EncoderReconstructiveSyDRA(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        # --- Block 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))

        # --- Block 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))

        # --- Block 3 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))

        # --- Block 4 ---
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))

        self.mgfd = MGFD(
            in_ch=base_width * 8, out_ch=base_width * 8
        )

        # --- Block 5 (Bottleneck) ---
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x);
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1);
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2);
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3);
        mp4 = self.mp4(b4)

        mp4_refined = self.mgfd(mp4)

        b5 = self.block5(mp4_refined)
        return b5


class SyDRAReconstructiveStream(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        super().__init__()
        self.encoder = EncoderReconstructiveSyDRA(in_channels, base_width)

        self.mccc = MCCC(channels=base_width * 8)

        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, x):
        b5 = self.encoder(x)

        # Apply contextual calibration to the latent reference.
        b5_fused = self.mccc(b5)

        output = self.decoder(b5_fused)
        return output

# ============================================================== #
# SyDRA discriminative encoder: DISA at the first two stages.
# ============================================================== #

class EncoderDiscriminativeSyDRA(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        # Layer 1 with DISA.
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
                                    nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.disa1 = DISA(
            base_width, num_heads=4, window_size=8
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))

        # Layer 2 with DISA.
        self.block2 = nn.Sequential(nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
                                    nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.disa2 = DISA(
            base_width * 2, num_heads=4, window_size=8
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))

        # Layer 3 has no DISA in the default configuration.
        self.block3 = nn.Sequential(nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
                                    nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))

        # Layer 4-6 (Standard)
        self.block4 = nn.Sequential(nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
                                    nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
                                    nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
                                    nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x);
        b1 = self.disa1(b1);
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1);
        b2 = self.disa2(b2);
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2);
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3);
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4);
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class SyDRADiscriminativeStream(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, out_features=False):
        super().__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminativeSyDRA(in_channels, base_width)
        # Standard discriminative decoder.
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        self.out_features = out_features

    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)
        output_segment = self.decoder_segment(b1, b2, b3, b4, b5, b6)
        if self.out_features:
            return output_segment, b2, b3, b4, b5, b6
        else:
            return output_segment

# ============================================================== #
# Reconstructive decoder.
# ============================================================== #
class DecoderReconstructive(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
                                 nn.Conv2d(base_width * 2, base_width * 1, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 1), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(nn.Conv2d(base_width * 1, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b5):
        up1 = self.up1(b5);
        db1 = self.db1(up1)
        up2 = self.up2(db1);
        db2 = self.db2(up2)
        up3 = self.up3(db2);
        db3 = self.db3(up3)
        up4 = self.up4(db3);
        db4 = self.db4(up4)
        out = self.fin_out(db4)
        return out


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()
        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
                                  nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(nn.Conv2d(base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(nn.Conv2d(base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
                                 nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(nn.Conv2d(base_width * (2 + 1), base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6);
        cat_b = torch.cat((up_b, b5), dim=1);
        db_b = self.db_b(cat_b)
        up1 = self.up1(db_b);
        cat1 = torch.cat((up1, b4), dim=1);
        db1 = self.db1(cat1)
        up2 = self.up2(db1);
        cat2 = torch.cat((up2, b3), dim=1);
        db2 = self.db2(cat2)
        up3 = self.up3(db2);
        cat3 = torch.cat((up3, b2), dim=1);
        db3 = self.db3(cat3)
        up4 = self.up4(db3);
        cat4 = torch.cat((up4, b1), dim=1);
        db4 = self.db4(cat4)
        return self.fin_out(db4)
