from model_zoo.convolutional_autoencoders import Encoder, Decoder
from model_zoo.deformer import *


class MorphAEus(nn.Module):
    """
        Paper 1367 - MICCAI.
        Code for deformable auto-encoders (MorphAEus)
            to learn pseudo-healthy  reconstructions
            and locally adapt their morphometry based on estimated dense deformation fields.
    """
    def __init__(self,
                 inshape,
                 in_channels: int, channels: Sequence[int], out_ch: int, strides: Sequence[int],
                 kernel_size=3, norm='batch', act='leakyrelu', deconv_mode='upsample', act_final='sigmoid',
                 bottleneck=False, skip=False, nr_ref_channels=2, bidir=True):
        """
        Parameters:
            inshape: Input shape. e.g. (1, 128, 128)
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
        """
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, channels=channels, strides=strides,
                              kernel_size=kernel_size, norm=norm, act=act, deconv_mode=deconv_mode, name_prefix='conv_')

        self.decoder = Decoder(in_channels=channels[-1], channels=channels, out_ch=out_ch,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, deconv_mode=deconv_mode,
                              act_final=act_final, bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='conv_')

        self.nr_ref_channels = nr_ref_channels
        self.nr_channels = len(channels)
        ref_channels = channels[:self.nr_ref_channels+1]

        self.deformer = Deformer(inshape=inshape, in_channels=in_channels, channels=ref_channels,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, deconv_mode=deconv_mode,
                                name_prefix='ref_', bidir=bidir)

    def forward(self, x, registration=False):
        encode_history, decode_history = [], []
        for i_e, enc_layer in enumerate(self.encoder):
            if i_e == 0:
                enc_x = enc_layer(x)
            else:
                enc_x = enc_layer(enc_x)
            if isinstance(enc_layer, nn.Conv2d):
                if enc_x.shape[-1] != 1:
                    encode_history.insert(0, enc_x)
        for i_d, dec_layer in enumerate(self.decoder):
            if i_d == 0:
                dec_x = dec_layer(enc_x)
            else:
                dec_x = dec_layer(dec_x)
            if isinstance(dec_layer, nn.Conv2d) or isinstance(dec_layer, nn.ConvTranspose2d):
                if len(decode_history) < len(encode_history) and \
                        encode_history[len(decode_history)].shape[-1] == dec_x.shape[-1]:
                    decode_history.append(dec_x)

        y_source, y_target, preint_flow, pos_flow = self.deformer(x, dec_x, encode_history, decode_history, registration)
        # return non-integrated flow field if training
        if not registration:
            return y_source, {'deformation': preint_flow, 'x_prior': dec_x, 'x_reversed': y_target}
        else:
            return y_source, {'deformation': pos_flow, 'x_prior': dec_x, 'x_reversed': y_source}
