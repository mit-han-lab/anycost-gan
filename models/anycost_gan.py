import math
import random

import torch
from torch import nn

from models.ops import *

G_CHANNEL_CONFIG = {
    4: 4096,
    8: 2048,
    16: 1024,
    32: 512,
    64: 256,
    128: 128,
    256: 64,
    512: 32,
    1024: 16,
}

D_CHANNEL_CONFIG = G_CHANNEL_CONFIG


class Generator(nn.Module):
    def __init__(
            self,
            resolution,
            style_dim=512,
            n_mlp=8,
            channel_multiplier=2,
            channel_max=512,
            blur_kernel=(1, 3, 3, 1),
            lr_mlp=0.01,  # reducing the learning rate of the style network
            act_func='lrelu',
    ):
        super().__init__()
        self.resolution = resolution
        self.style_dim = style_dim  # usually 512
        self.channel_max = channel_max

        style_mlp = [EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='lrelu') for _ in range(n_mlp)]
        style_mlp.insert(0, PixelNorm())
        self.style = nn.Sequential(*style_mlp)

        self.channels = {k: min(channel_max, int(v * channel_multiplier)) for k, v in G_CHANNEL_CONFIG.items()}

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, activation=act_func,
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_res = int(math.log(resolution, 2))
        self.num_layers = (self.log_res - 2) * 2 + 1
        self.n_style = self.log_res * 2 - 2

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_channel = self.channels[4]
        for i in range(3, self.log_res + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel,
                                         activation=act_func))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, activation=act_func))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        self.noises = nn.Module()
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

    def make_noise(self):
        device = self.style[-1].weight.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_res + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def mean_style(self, n_sample):
        z = torch.randn(n_sample, self.style_dim, device=self.style[-1].weight.device)
        w = self.style(z).mean(0, keepdim=True)
        return w

    def get_style(self, z):
        z_shape = z.shape
        return self.style(z.view(-1, z.shape[-1])).view(z_shape)

    def forward(
            self,
            styles,
            return_styles=False,
            inject_index=None,
            truncation=1,
            truncation_style=None,
            input_is_style=False,
            noise=None,
            randomize_noise=True,
            return_rgbs=False,
            target_res=None,
    ):
        """
        :param styles: the input z or w, depending on input_is_style arg
        :param return_styles: whether to return w (used for training)
        :param inject_index: manually assign injection index
        :param truncation: whether to apply style truncation. default: no truncate
        :param truncation_style: the mean style used for truncation
        :param input_is_style: whether the input is style (w) or z
        :param noise: manually assign noise tensor per layer
        :param randomize_noise: whether to randomly draw the noise or use the fixed noise
        :param return_rgbs: whether to return all the lower resolution outputs
        :param target_res: assign target resolution; rarely used here
        :return: output image, _
        """
        # 1. get the style code (i.e., w+)
        assert len(styles.shape) == 3  # n, n_lat, lat_dim
        if not input_is_style:  # map from z to w
            styles = self.get_style(styles)

        if truncation < 1:
            styles = (1 - truncation) * truncation_style.view(1, 1, -1) + truncation * styles

        if styles.shape[1] == 1:  # only one style provided
            styles = styles.repeat(1, self.n_style, 1)
        elif styles.shape[1] == 2:  # two styles to mix
            if inject_index is None:
                inject_index = random.randint(1, self.n_style - 1)
            style1 = styles[:, 0:1].repeat(1, inject_index, 1)
            style2 = styles[:, 1:2].repeat(1, self.n_style - inject_index, 1)
            styles = torch.cat([style1, style2], 1)
        else:  # full style
            assert styles.shape[1] == self.n_style

        # 2. get noise
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        # 3. generate images
        all_rgbs = []

        out = self.input(styles.shape[0])  # get constant input
        out = self.conv1(out, styles[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, styles[:, 1])
        all_rgbs.append(skip)

        if hasattr(self, 'target_res') and target_res is None:  # a quick fix for search
            target_res = self.target_res

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, styles[:, i], noise=noise1)
            out = conv2(out, styles[:, i + 1], noise=noise2)
            skip = to_rgb(out, styles[:, i + 2], skip)
            all_rgbs.append(skip)

            i += 2
            if target_res is not None and skip.shape[-1] == target_res:
                break

        if return_styles:
            return skip, styles
        elif return_rgbs:
            return skip, all_rgbs
        else:
            return skip, None


class Discriminator(nn.Module):
    def __init__(self, resolution, channel_multiplier=2, channel_max=512, blur_kernel=(1, 3, 3, 1), act_func='lrelu'):
        super().__init__()
        # raise NotImplementedError

        channels = {
            4: 4096,
            8: 2048,
            16: 1024,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16,
        }

        channels = {k: min(channel_max, int(v * channel_multiplier)) for k, v in channels.items()}

        convs = [ConvLayer(3, channels[resolution], 1, activate=act_func)]

        log_res = int(math.log(resolution, 2))

        in_channel = channels[resolution]

        for i in range(log_res, 2, -1):
            out_channel = channels[2 ** (i - 1)]  # the out channel corresponds to a lower resolution
            convs.append(ResBlock(in_channel, out_channel, blur_kernel, act_func=act_func))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, activate=act_func)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation=act_func),
            EqualLinear(channels[4], 1),
        )

    def forward(self, x):
        out = self.convs(x)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class DiscriminatorMultiRes(nn.Module):
    def __init__(self, resolution, channel_multiplier=2, channel_max=512, blur_kernel=(1, 3, 3, 1), act_func='lrelu',
                 n_res=1, modulate=False):
        super().__init__()

        channels = {k: min(channel_max, int(v * channel_multiplier)) for k, v in D_CHANNEL_CONFIG.items()}

        self.convs = nn.ModuleList()
        self.res2idx = {}  # a mapping to find the right conv to map the input image
        for i_res in range(n_res):
            cur_res = resolution // (2 ** i_res)
            self.res2idx[cur_res] = i_res
            self.convs.append(ConvLayer(3, channels[cur_res], 1, activate=act_func))

        log_res = int(math.log(resolution, 2))
        in_channel = channels[resolution]

        self.blocks = nn.ModuleList()
        for i in range(log_res, 2, -1):
            out_channel = channels[2 ** (i - 1)]  # the out channel corresponds to a lower resolution
            self.blocks.append(
                ResBlock(in_channel, out_channel, blur_kernel, act_func=act_func,
                         modulate=modulate and i in list(range(log_res, 2, -1))[-2:],  # add g_arch modulation
                         g_arch_len=4 * (log_res * 2 - 2))
            )
            in_channel = out_channel

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, activate=act_func)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation=act_func),
            EqualLinear(channels[4], 1),
        )

    def forward(self, x, g_arch=None):
        res = x.shape[-1]
        idx = self.res2idx[res]
        out = self.convs[idx](x)
        for i in range(idx, len(self.blocks)):
            out = self.blocks[i](out, g_arch)

        out = self.minibatch_discrimination(out, self.stddev_group, self.stddev_feat)
        out = self.final_conv(out).view(out.shape[0], -1)
        out = self.final_linear(out)

        return out

    @staticmethod
    def minibatch_discrimination(x, stddev_group, stddev_feat):
        out = x
        batch, channel, height, width = out.shape
        group = min(batch, stddev_group)
        stddev = out.view(group, -1, stddev_feat, channel // stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        return out
