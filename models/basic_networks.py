"""
    Some basci net module class.

"""
import torch
import torch.nn as nn
import functools
import numpy as np
import math
import torch.nn.functional as F

def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):

    def __init__(self,
                 rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)  # inchannel 3 outchannl 3, F=1
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    """
        conv -> relu -> conv
    """
    def __init__(self,
                 conv,
                 n_feats,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResnetGenerator(nn.Module):
    """
        Defines the generator that consists of Resnet blocks between a few
        downsampling/upsampling operations.
        Code and idea originally from Justin Johnson's architecture.
        https://github.com/jcjohnson/fast-neural-style/
    """
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks=6,
                 gpu_ids=[],
                 use_parallel=True,
                 learn_residual=False,
                 padding_type='reflect'):

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [                                           # shape after padding, exampel pic shape 256 x 256
            nn.ReflectionPad2d(3),                          # S x 3   @ (262) x (262)
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),  # S x 64  @ (256) x (256)
            norm_layer(ngf),                                # S x 64  @ (256) x (256)
            nn.ReLU(True)                                   # S x 64  @ (256) x (256)
        ]

        # down sampling

        model += [                                                                  # Input: S x 64  @ 256 x 256
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),  # S x 128 @ 128 x 128
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias), # S x 256 @ 64 x 64
            norm_layer(256),
            nn.ReLU(True)
        ]

        # the middle ResBlocks
        for i in range(n_blocks):
            model += [ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)] # S x 256 @ 64 x 64


        # H out ​ =(H in ​ −1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        model += [                                                                                              # S x 256 @ 64 x 64
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),  # S x 128 @ 128 x 128
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),  # S x  64  @ 256 x 256
            norm_layer(64),
            nn.ReLU(True),
        ]

        model += [
            nn.ReflectionPad2d(3),                              # S x 64 @ 262 x 262
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0), # S x 3  @ 256 x 256
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self,  input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output


class ResnetBlock(nn.Module):
    """
         Define a resnet block
    """
    def __init__(self,
                 dim,
                 padding_type,
                 use_dropout,
                 use_bias,
                 norm_layer=nn.BatchNorm2d):
        """
          init function
        :param dim: input channel
        :param padding_type: padding method, reflect, replicate, zero
        :param norm_layer: thre nrom layer
        :param use_dropout: use or not
        :param use_bias: use or not
        """
        super(ResnetBlock, self).__init__()

        padAndConv = {
            'reflect':
                [nn.ReflectionPad2d(1),
                 nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'replicate':
                [nn.ReplicationPad2d(1),
                 nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'zero':
                [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
        }

        try:
            blocks = padAndConv[padding_type] + [norm_layer(dim), nn.ReLU(True)] + [nn.Dropout(0.5)]\
                if use_dropout else [] + padAndConv[padding_type] + [norm_layer(dim)]
        except:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetGenerator(nn.Module):
    """
        Defines the Unet generator.
        |num_downs|: number of downsamplings in UNet. For example,
        if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
    """
    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 gpu_ids=[],
                 use_parallel=True,
                 learn_residual=False):

        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


class UnetSkipConnectionBlock(nn.Module):
    """
        Defines the submodule with skip connection.
        X -------------------identity---------------------- X
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self,
                 outer_nc,
                 inner_nc,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False):

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dRelu = nn.LeakyReLU(0.2, True)
        dNorm = norm_layer(inner_nc)
        uRelu = nn.ReLU(True)
        uNorm = norm_layer(outer_nc)

        if outermost:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dConv]
            uModel = [uRelu, uConv, nn.Tanh()]
            model = [
                dModel,
                submodule,
                uModel
            ]
        # model = [
        # 	# Down
        # 	nn.Conv2d( outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        #
        # 	submodule,
        # 	# Up
        # 	nn.ReLU(True),
        # 	nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
        # 	nn.Tanh()
        # ]
        elif innermost:
            uConv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv, uNorm]
            model = [
                dModel,
                uModel
            ]
        # model = [
        # 	# down
        # 	nn.LeakyReLU(0.2, True),
        # 	# up
        # 	nn.ReLU(True),
        # 	nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        # 	norm_layer(outer_nc)
        # ]
        else:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv, dNorm]
            uModel = [uRelu, uConv, uNorm]

            model = [
                dModel,
                submodule,
                uModel
            ]
            model += [nn.Dropout(0.5)] if use_dropout else []

        # if use_dropout:
        # 	model = down + [submodule] + up + [nn.Dropout(0.5)]
        # else:
        # 	model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


class NLayerDiscriminator(nn.Module):
    """
        Defines the PatchGAN discriminator with the specified arguments.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [                                                              # S x 3  @ 256 x 256
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),     # S x 64 @ 129 x 129
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
