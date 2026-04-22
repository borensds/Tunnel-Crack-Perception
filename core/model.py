import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class DLKA_Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()
        )
        self.lka = LKA(out_channels)

    def forward(self, x):
        x = self.proj(x)
        return self.lka(x) + x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MDCNet(nn.Module):
    def __init__(self, num_classes=1, mode='ours'):
        super().__init__()
        print(f"初始化 MDCNet, 模式: {mode}")

        self.encoder = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()
        current_channels = enc_channels[-1]

        use_dlka = 'no_dlka' not in mode
        use_coord = 'no_coord' not in mode

        # Bridge
        if use_dlka:
            self.d_lka = DLKA_Bottleneck(current_channels, 256)
        else:
            self.d_lka = nn.Sequential(
                nn.Conv2d(current_channels, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.Hardswish()
            )

        # Attention
        self.coord_att = CoordAtt(256, 256) if use_coord else nn.Identity()

        # Decoder
        self.up1 = DecoderBlock(256, enc_channels[-2], 128)
        self.up2 = DecoderBlock(128, enc_channels[-3], 64)
        self.up3 = DecoderBlock(64, enc_channels[-4], 32)
        self.up4 = DecoderBlock(32, enc_channels[-5], 16)

        # Head
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]

        x = self.d_lka(x)
        x = self.coord_att(x)

        x = self.up1(x, features[-2])
        x = self.up2(x, features[-3])
        x = self.up3(x, features[-4])
        x = self.up4(x, features[-5])

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.final_conv(x)


if __name__ == "__main__":
    model = MDCNet(mode='ours')
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)
    print(f"模型构建成功！总参数大小: {param_size:.2f} MB")

    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"推理测试通过！输出形状: {output.shape}")