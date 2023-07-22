import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Sequence
from torch.distributions.normal import Normal
from net_utils.activation_layers import CustomSwish


class Deformer(nn.Module):
    def __init__(self, inshape, channels: Sequence[int], strides: Sequence[int], in_channels=1,
                 kernel_size: int = 3, norm: str = 'batch', act: str = 'swish', deconv_mode='trans',
                 name_prefix: str = '_',
                 int_steps=7,
                 int_downsize=1,
                 bidir=False,
                 unet_half_res=False,):
        """
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        :param add_final:
        :param name_prefix:
        """
        super(Deformer, self).__init__()
        ndims = len(inshape)
        padding = (kernel_size - 1) // 2
        encode_channel_list = list(reversed(channels))[1:]
        decode_channel_list = list(reversed(channels[1:]))
        decode_strides = strides[::-1] or [1]

        self.deformerNN = nn.Sequential()
        for i, (c, s) in enumerate(zip(decode_channel_list, decode_strides)):
            layer_channels = decode_channel_list[i] + encode_channel_list[i]
            layer_channels = layer_channels + 32 if i > 0 else layer_channels

            if deconv_mode == 'upsample':
                # if i > 0:
                self.deformerNN.add_module(name_prefix + "_upsample_%i" % i,
                                           nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True))
                self.deformerNN.add_module(name_prefix + "_refine_%i" % i,
                                           nn.Conv2d(in_channels=layer_channels, out_channels=32,
                                          kernel_size=(kernel_size, kernel_size)
                                          , padding=padding))
            else:
                self.deformerNN.add_module(name_prefix + "_refine_%i" % i,
                                           nn.ConvTranspose2d(in_channels=layer_channels, out_channels=32,
                                                   kernel_size=(kernel_size, kernel_size), stride=s,
                                                   padding=padding,
                                                   output_padding=1))
            if norm == 'batch':
                self.deformerNN.add_module(name_prefix + "_batch_%i" % i, nn.BatchNorm2d(32))

            if act == 'relu':
                self.deformerNN.add_module(name_prefix + "_act_%i" % i, nn.ReLU(True))
            elif act =='leakyrelu':
                self.deformerNN.add_module(name_prefix + "_act_%i" % i, nn.LeakyReLU(0.2))
            else:
                self.deformerNN.add_module(name_prefix + "_act_%i" % i, CustomSwish())

        self.flow = nn.Conv2d(32, ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        self.nr_layers = len(decode_channel_list)

    def forward(self, x, dec_x, encode_history, decode_history, registration):
        i_r_level = len(encode_history) - self.nr_layers
        assert i_r_level >= 0, 'No history for deformation estimation.'
        for i_r, ref_layer in enumerate(self.deformerNN):
            if i_r == 0:
                disp_x = ref_layer(torch.cat([encode_history[i_r_level], decode_history[i_r_level]], dim=1))
                i_r_level += 1
            else:
                if isinstance(ref_layer, nn.Conv2d) or isinstance(ref_layer, nn.ConvTranspose2d):
                    disp_x = ref_layer(torch.cat([encode_history[i_r_level], decode_history[i_r_level], disp_x], dim=1))
                    i_r_level += 1
                else:
                    disp_x = ref_layer(disp_x)

        flow_field = self.flow(disp_x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None
            #
            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(dec_x, pos_flow)
        y_target = self.transformer(x, neg_flow) if self.bidir else y_source
        #
        return y_source, y_target, preint_flow, pos_flow


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    code from https://github.com/voxelmorph/voxelmorph

    License:
        Apache License 2.0
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.

    code from https://github.com/voxelmorph/voxelmorph

    License:
        Apache License 2.0
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.

    code from https://github.com/voxelmorph/voxelmorph

    License:
        Apache License 2.0
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x