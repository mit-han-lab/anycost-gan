import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

try:  # CUDA kernel
    assert not ('FORCE_NATIVE' in os.environ and os.environ['FORCE_NATIVE'])  # add FORCE_NATIVE in env to force native
    from cuda_op.fused_act import FusedLeakyReLU, fused_leaky_relu
    from cuda_op.upfirdn2d import upfirdn2d
except Exception as e:
    print(e)
    print(' # Using native op...')
    from cuda_op.op_native import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

__all__ = ['PixelNorm', 'EqualConv2d', 'EqualLinear', 'ModulatedConv2d', 'StyledConv', 'ConvLayer', 'ResBlock',
           'ConstantInput', 'ToRGB']


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)

        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:  # support dynamic channel
            assert self.first_k_oup <= out.shape[1]
            return out[:, :self.first_k_oup]
        else:
            return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()  # random noise

        return image + self.weight * noise


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()
    k = torch.flip(k, [0, 1])  # move from runtime to here
    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, x):
        in_channel = x.shape[1]
        weight = self.weight

        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:
            weight = weight[:self.first_k_oup]

        weight = weight[:, :in_channel].contiguous()  # index sub channels for inference

        out = F.conv2d(
            x,
            weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1., activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            if self.activation == 'lrelu':
                out = fused_leaky_relu(out, self.bias * self.lr_mul)
            else:
                raise NotImplementedError

        else:
            out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample  # if true, use deconvolution
        self.downsample = downsample
        assert not downsample, 'Downsample is not implemented yet!'
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2 + factor - 1, p // 2 + 1), upsample_factor=factor)

        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape

        style = self.modulation(style)
        style = style.view(batch, 1, -1, 1, 1)

        # process weight for dynamic channel
        first_k_oup = self.first_k_oup if hasattr(self, 'first_k_oup') and self.first_k_oup is not None \
            else self.weight.shape[1]
        assert first_k_oup <= self.weight.shape[1]

        weight = self.weight
        weight = weight[:, :first_k_oup, :in_channel].contiguous()  # index sub channels fro inference
        # modulate weight
        weight = self.scale * weight * style[:, :, :in_channel]

        # demodulate weight
        if self.demodulate:
            weight = weight * torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + self.eps)

        if self.upsample:
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.transpose(1, 2)
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2], weight.shape[3],
                                    weight.shape[4])
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            out = out.view(batch, -1, out.shape[-2], out.shape[-1])
            out = self.blur(out)
        else:
            x = x.contiguous().view(1, batch * in_channel, height, width)
            weight = weight.view(weight.shape[0] * weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4])
            out = F.conv2d(x, weight, padding=self.padding, groups=batch)
            out = out.view(batch, -1, out.shape[-2], out.shape[-1])

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=(1, 3, 3, 1),
            demodulate=True,
            activation='lrelu',
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        if activation == 'lrelu':
            self.activate = FusedLeakyReLU(out_channel)
        else:
            raise NotImplementedError

    def forward(self, x, style, noise=None):
        out = self.conv(x, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class AdaptiveModulate(nn.Module):
    def __init__(self, num_features, g_arch_len):
        super(AdaptiveModulate, self).__init__()
        self.weight_mapping = nn.Linear(g_arch_len, num_features)
        self.bias_mapping = nn.Linear(g_arch_len, num_features)

    def forward(self, x, g_arch):
        assert x.dim() == 4
        weight = self.weight_mapping(g_arch.view(1, -1)).view(-1) + 1.  # add 1 to make a smooth start
        bias = self.bias_mapping(g_arch.view(1, -1)).view(-1)
        return x * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=(1, 3, 3, 1),
            bias=True,
            activate='lrelu',
            modulate=False,
            g_arch_len=18 * 4,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(in_channel,
                        out_channel,
                        kernel_size,
                        padding=self.padding,
                        stride=stride,
                        bias=bias and not activate,
                        )
        )

        # if conditioned on g_arch
        if modulate:
            layers.append(AdaptiveModulate(out_channel, g_arch_len))

        assert bias == (activate != 'none')
        if activate == 'lrelu':  # if activate then bias = True
            layers.append(FusedLeakyReLU(out_channel))
        else:
            assert activate == 'none'

        super().__init__(*layers)

    def forward(self, x, g_arch=None):
        for module in self:
            if isinstance(module, AdaptiveModulate):
                x = module(x, g_arch)
            else:
                x = module(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=(1, 3, 3, 1), act_func='lrelu',
                 modulate=False, g_arch_len=18 * 4):
        super().__init__()
        self.out_channel = out_channel
        self.conv1 = ConvLayer(in_channel, in_channel, 3, activate=act_func, modulate=modulate, g_arch_len=g_arch_len)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, blur_kernel=blur_kernel, activate=act_func,
                               modulate=modulate, g_arch_len=g_arch_len)

        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate='none', bias=False,
                              modulate=modulate, g_arch_len=g_arch_len)

    def forward(self, x, g_arch=None):
        out = self.conv1(x, g_arch)
        out = self.conv2(out, g_arch)

        skip = self.skip(x, g_arch)
        out = (out + skip) / math.sqrt(2)

        return out
