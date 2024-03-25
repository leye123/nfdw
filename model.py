import torch
import numbers
import torch.nn.functional as TF
from torch import nn
import numpy as np
from mamba_ssm import Mamba
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
    

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class LatentPixelAttention(nn.Module):
    def __init__(self, dim, bias):
        super(LatentPixelAttention, self).__init__()
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = TF.normalize(q, dim=-1)
        k = TF.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k)
        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = rearrange(out, 'b c (h w) -> b c h w', h=int(h),
                        w=int(w))

        out = self.project_out(out)

        return out
    

class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * 3)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = TF.relu(x1) * x2
        x = self.project_out(x)
        return x
    

class LatentPixelTransformerBlock(nn.Module):
    def __init__(self, dim, bias=False):
        super(LatentPixelTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = LatentPixelAttention(dim, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = TF.pad(y, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = x + y
        x = x + self.ffn(self.norm2(x))

        return x
    

class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = TF.normalize(q, dim=-1)
        k = TF.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(int(c / self.num_heads))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

class CATransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        super(CATransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = ChannelAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = TF.pad(y, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = x + y
        x = x + self.ffn(self.norm2(x))

        return x
    

def window_partition(x, window_size: int, h, w):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - w % window_size) % window_size
    pad_b = (window_size - h % window_size) % window_size
    x = TF.pad(x, [pad_l, pad_r, pad_t, pad_b])
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    H = H + pad_b
    W = W + pad_r
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    windows = TF.pad(x, [pad_l, -pad_r, pad_t, -pad_b])
    return windows


class SWPSA(nn.Module):
    def __init__(self, dim, window_size, shift_size, bias):
        super(SWPSA, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

        self.qkv_conv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

    def window_partitions(self, x, window_size: int):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size(M)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def create_mask(self, x):

        n, c, H, W = x.shape
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partitions(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape

        x = window_partition(x, self.window_size, h, w)

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q.transpose(-2, -1) @ k) / self.window_size
        attn = attn.softmax(dim=-1)
        out = (v @ attn)
        out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        w=int(self.window_size))
        out = self.project_out(out)
        out = window_reverse(out, self.window_size, h, w)

        shift = torch.roll(out, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        shift_window = window_partition(shift, self.window_size, h, w)
        qkv = self.qkv_dwconv1(self.qkv_conv1(shift_window))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) / self.window_size
        mask = self.create_mask(shortcut)
        attn = attn.view(b, -1, self.window_size * self.window_size,
                         self.window_size * self.window_size) + mask.unsqueeze(0)
        attn = attn.view(-1, self.window_size * self.window_size, self.window_size * self.window_size)
        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        w=int(self.window_size))

        out = self.project_out1(out)
        out = window_reverse(out, self.window_size, h, w)
        out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return out


class SWPSATransformerBlock(nn.Module):
    def __init__(self, dim, window_size=8, shift_size=3, bias=False):
        super(SWPSATransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = SWPSA(dim, window_size, shift_size, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = TF.pad(y, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = x + y
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, window_size=8, shift_size=3, bias=False):
        super(TransformerBlock, self).__init__()

        self.globa = CATransformerBlock(dim, num_heads, bias)

        self.pixel = SWPSATransformerBlock(dim, window_size, shift_size, bias)

        self.alpha = nn.Parameter(torch.ones(1) / 2)

    def forward(self, x):
        x = self.alpha * self.pixel(x) + (1 - self.alpha) * self.globa(x)

        return x


class LatentTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        super(LatentTransformerBlock, self).__init__()

        self.csa = CATransformerBlock(dim, num_heads, bias)

        self.pixel = LatentPixelTransformerBlock(dim, bias)

        self.belta = nn.Parameter(torch.ones(1) / 2)

    def forward(self, x):
        x = self.belta * self.pixel(x) + (1 - self.belta) * self.csa(x)

        return x
    

class UVMB(nn.Module):
    def __init__(self,c=3,w=256,h=256):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=1),
            LatentTransformerBlock(dim=16),
            TransformerBlock(dim=16),
            nn.Conv2d(in_channels=16, out_channels=c, kernel_size=3, stride=1, padding=1)
        )

        self.model1 = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.model3 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=w*h, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)

        self.ln = nn.LayerNorm(normalized_shape=c)

    def forward(self, x):
        b, c, w, h = x.shape
        x = self.trans(x) + x
        x = self.ln(x.reshape(b, -1, c))
        y = self.model1(x).permute(0, 2, 1)
        z = self.model3(y).permute(0, 2, 1)
        att = self.model2(x).softmax(dim=-1)
        result = att * z
        output = result.reshape(b, c, w, h)
        return self.smooth(output)
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.ub = UVMB(c=in_channels, w=64, h=64)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        inputs = TF.interpolate(x, size=[64, 64], mode='bilinear', align_corners=True)
        outputs = self.ub(inputs)
        outputs = TF.interpolate(outputs, size=[x.shape[2], x.shape[3]],   mode='bilinear', align_corners=True) + x
        return self.double_conv(outputs)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = TF.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    


class Model(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(Model, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)

    def forward(self, inp):
        x = inp
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) + inp
        return x



if __name__ == '__main__':
    with torch.no_grad():
        t = torch.randn(1, 3, 600, 400).cuda()
        model = Model().cuda()
        out = model(t)
        print(out.shape)
