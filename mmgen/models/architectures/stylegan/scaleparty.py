import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgen.models.architectures import PixelNorm, positional_encoding
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module
from .modules.styleganv2_modules import (ConstantInput, ConvDownLayer,
                                         EqualLinearActModule,
                                         ModMBStddevLayer,
                                         ModulatedPEStyleConv, ModulatedToRGB,
                                         ResBlock)
from .utils import get_mean_latent, style_mixing


@MODULES.register_module()
class ScalePartyGenerator(nn.Module):
    """StyleGAN2 Generator.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of convolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    Args:
        out_size (int): The output size of the StyleGAN2 generator.
        style_channels (int): The number of channels for style code.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probability. The value should be
            in range of [0, 1]. Defaults to 0.9.
    """

    def __init__(self,
                 out_size,
                 style_channels,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 no_pad=False,
                 deconv2conv=False,
                 interp_pad=None,
                 up_config=dict(scale_factor=2, mode='nearest'),
                 up_after_conv=False,
                 head_pos_encoding=None,
                 head_pos_size=(4, 4),
                 interp_head=False,
                 use_noise=True):
        super().__init__()
        self.out_size = out_size
        self.style_channels = style_channels
        self.num_mlps = num_mlps
        self.channel_multiplier = channel_multiplier
        self.lr_mlp = lr_mlp
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode
        self.mix_prob = mix_prob
        self.no_pad = no_pad
        self.deconv2conv = deconv2conv
        self.interp_pad = interp_pad
        self.with_interp_pad = interp_pad is not None
        self.up_config = deepcopy(up_config)
        self.up_after_conv = up_after_conv
        self.head_pos_encoding = head_pos_encoding
        self.head_pos_size = head_pos_size
        self.interp_head = interp_head
        self.use_noise = use_noise

        # define style mapping layers
        mapping_layers = [PixelNorm()]

        for _ in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    style_channels,
                    style_channels,
                    equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.),
                    act_cfg=dict(type='fused_bias')))

        self.style_mapping = nn.Sequential(*mapping_layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        in_ch = self.channels[4]
        # constant input layer
        if self.head_pos_encoding:
            if self.head_pos_encoding['type'] in [
                    'CatersianGrid', 'CSG', 'CSG2d', 'ScaleParty'
            ]:
                in_ch = 2
            self.head_pos_enc = build_module(self.head_pos_encoding)
        else:
            size_ = 4
            if self.no_pad:
                size_ += 2
            self.constant_input = ConstantInput(self.channels[4], size=size_)

        # 4x4 stage
        self.conv1 = ModulatedPEStyleConv(
            in_ch,
            self.channels[4],
            kernel_size=3,
            style_channels=style_channels,
            blur_kernel=blur_kernel,
            deconv2conv=self.deconv2conv,
            no_pad=self.no_pad,
            up_config=self.up_config,
            interp_pad=self.interp_pad,
            use_noise=use_noise)
        self.to_rgb1 = ModulatedToRGB(
            self.channels[4], style_channels, upsample=False, clip=True)

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channels_ = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channels_ = self.channels[2**i]

            self.convs.append(
                ModulatedPEStyleConv(
                    in_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    deconv2conv=self.deconv2conv,
                    no_pad=self.no_pad,
                    up_config=self.up_config,
                    interp_pad=self.interp_pad,
                    up_after_conv=self.up_after_conv,
                    use_noise=use_noise))
            self.convs.append(
                ModulatedPEStyleConv(
                    out_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=False,
                    blur_kernel=blur_kernel,
                    deconv2conv=self.deconv2conv,
                    no_pad=self.no_pad,
                    up_config=self.up_config,
                    interp_pad=self.interp_pad,
                    up_after_conv=self.up_after_conv,
                    use_noise=use_noise))
            self.to_rgbs.append(
                ModulatedToRGB(out_channels_, style_channels, upsample=True, clip=True))
                

            in_channels_ = out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1

        # register buffer for injected noises
        noises = self.make_injected_noise()
        for layer_idx in range(self.num_injected_noises):
            self.register_buffer(f'injected_noise_{layer_idx}',
                                 noises[layer_idx])

    def train(self, mode=True):
        if mode:
            if self.default_style_mode != self._default_style_mode:
                mmcv.print_log(
                    f'Switch to train style mode: {self._default_style_mode}',
                    'mmgen')
            self.default_style_mode = self._default_style_mode

        else:
            if self.default_style_mode != self.eval_style_mode:
                mmcv.print_log(
                    f'Switch to evaluation style mode: {self.eval_style_mode}',
                    'mmgen')
            self.default_style_mode = self.eval_style_mode

        return super(ScalePartyGenerator, self).train(mode)

    def make_injected_noise(self, chosen_scale=0):

        device = get_module_device(self)

        base_scale = 2**2 + 4 + chosen_scale
        logsize = int(np.log2(256))

        noises = [torch.randn(1, 1, base_scale, base_scale, device=device)]
        size = base_scale 

        for i in range(3, logsize + 1):
            size = size * 2
            size = size - 2
            noises.append(
                    torch.randn(
                        1,
                        1,
                        size,
                        size,
                        device=device))

            size = size - 2

            noises.append(
                    torch.randn(
                        1,
                        1,
                        size,
                        size,
                        device=device))

        return noises

        # base_scale = 2**2 + chosen_scale

        # noises = [torch.randn(1, 1, base_scale, base_scale, device=device)]

        # for i in range(3, self.log_size + 1):
        #     for n in range(2):
        #         _pad = 0
        #         if self.no_pad and not self.up_after_conv and n == 0:
        #             _pad = 2
        #         noises.append(
        #             torch.randn(
        #                 1,
        #                 1,
        #                 base_scale * 2**(i - 2) + _pad,
        #                 base_scale * 2**(i - 2) + _pad,
        #                 device=device))

        # return noises

    def get_noise(self, num_batches):
        device = get_module_device(self)
        assert num_batches > 0 
        if self.default_style_mode == 'mix' and random.random(
        ) < self.mix_prob:
            styles = [
                torch.randn((num_batches, self.style_channels))
                for _ in range(2)
            ]
        else:
            styles = [torch.randn((num_batches, self.style_channels))]
        styles = [s.to(device) for s in styles]
        return styles


    def get_mean_latent(self, num_samples=4096, **kwargs):
        """Get mean latent of W space in this generator.

        Args:
            num_samples (int, optional): Number of sample times. Defaults
                to 4096.

        Returns:
            Tensor: Mean latent of this generator.
        """
        return get_mean_latent(self, num_samples, **kwargs)

    def style_mixing(self,
                     n_source,
                     n_target,
                     inject_index=1,
                     truncation_latent=None,
                     truncation=0.7,
                     chosen_scale=0):
        return style_mixing(
            self,
            n_source=n_source,
            n_target=n_target,
            inject_index=inject_index,
            truncation_latent=truncation_latent,
            truncation=truncation,
            style_channels=self.style_channels,
            chosen_scale=chosen_scale)

    def forward(self,
                styles,
                num_batches=-1,
                return_noise=False,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                injected_noise=None,
                randomize_noise=True,
                chosen_scale=0,
                positional_enc=None,
                double_eval=False,
                out_res=0):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torch.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered aoise tensor injected to the style conv
                block. Defaults to True.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
        """

        if double_eval:
            out_small = self(
                styles,
                num_batches,
                return_noise,
                True, # return latents
                inject_index,
                truncation,
                truncation_latent,
                input_is_latent,
                injected_noise,
                randomize_noise,
                chosen_scale=0,
                positional_enc=None,
                double_eval=False)

            img_small = out_small["fake_img"]
            styles = out_small["noise_batch"]

            img_large = self(
                styles,
                num_batches,
                False, # return noise
                False, # return latents
                inject_index,
                truncation,
                truncation_latent,
                input_is_latent, #input is latent
                injected_noise,
                randomize_noise,
                chosen_scale=2,
                positional_enc=None,
                double_eval=False)

            return [img_small, img_large]


        # receive noise and conduct sanity check.
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == self.style_channels
            styles = [styles]
        elif mmcv.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == self.style_channels
        # receive a noise generator and sample noise.
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    noise_generator((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [noise_generator((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]
        # otherwise, we will adopt default noise sampler.
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    torch.randn((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]

        if not input_is_latent:
            noise_batch = styles
            styles = [self.style_mapping(s) for s in styles]
        else:
            noise_batch = None

        # if isinstance(chosen_scale, list):
        #     assert not return_latents and not return_noise
        # else:
        #     chosen_scale = [chosen_scale]

        
        # imgs = []
        # for chosen_scale_ in chosen_scale: 
        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            elif chosen_scale > 0:
                if not hasattr(self, f'injected_noise_{chosen_scale}_0'):
                    noises_ = self.make_injected_noise(chosen_scale)
                    for i in range(self.num_injected_noises):
                        setattr(self, f'injected_noise_{chosen_scale}_{i}',
                                noises_[i])
                injected_noise = [
                    getattr(self, f'injected_noise_{chosen_scale}_{i}')
                    for i in range(self.num_injected_noises)
                ]
            else:
                injected_noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]
        # use truncation trick
        if truncation < 1:
            style_t = []
            # calculate truncation latent on the fly
            if truncation_latent is None and not hasattr(
                    self, 'truncation_latent'):
                self.truncation_latent = self.get_mean_latent()
                truncation_latent = self.truncation_latent
            elif truncation_latent is None and hasattr(self,
                                                    'truncation_latent'):
                truncation_latent = self.truncation_latent

            for style in styles:
                style_t.append(truncation_latent + truncation *
                            (style - truncation_latent))

            styles = style_t
        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if isinstance(chosen_scale, int):
            chosen_scale = (chosen_scale, chosen_scale)

        # 4x4 stage
        if self.head_pos_encoding:
            if positional_enc is not None:
                out = positional_enc
            elif out_res:
                step_size = self.out_size/self.head_pos_size[0]
                out_scale = self.head_pos_size[0]
                while (step_size * out_scale) < out_res:
                    out_scale += 1

                pos_len = (step_size * out_scale)/out_res
                # lenght = round((step_size * out_scale)/out_res,1)

                positional_enc = self.head_pos_enc.make_grid2d(
                            out_scale,
                            out_scale, 
                            latent.size(0),
                            start_h= 0.5 - pos_len/2,
                            start_w= 0.5 - pos_len/2,
                            width=  pos_len,
                            height= pos_len)
                out = positional_enc

            elif self.interp_head:
                out = self.head_pos_enc.make_grid2d(self.head_pos_size[0],
                                                    self.head_pos_size[1],
                                                    latent.size(0))
                h_in = self.head_pos_size[0] + chosen_scale[0]
                w_in = self.head_pos_size[1] + chosen_scale[1]
                out = F.interpolate(
                    out,
                    size=(h_in, w_in),
                    mode='bilinear',
                    align_corners=True)
            else:
                positional_enc = self.head_pos_enc.make_grid2d(
                    self.head_pos_size[0] + chosen_scale[0],
                    self.head_pos_size[1] + chosen_scale[1], latent.size(0))
                out = positional_enc
            out = out.to(latent)
        else:
            out = self.constant_input(latent)
            if chosen_scale[0] != 0 or chosen_scale[1] != 0:
                out = F.interpolate(
                    out,
                    size=(out.shape[2] + chosen_scale[0],
                        out.shape[3] + chosen_scale[1]),
                    mode='bilinear',
                    align_corners=True)

        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbs):
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)

            _index += 2

        img = skip[:,:,2:-2,2:-2]

        if out_res:
            start = (img.shape[-1] - out_res)//2
            img = img[:,:, start:start+out_res, start:start+out_res]
    

        # if len(chosen_scale) == 1:
        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch,
                injected_noise=injected_noise,
                positional_enc=positional_enc)
            return output_dict

        return img

        #     else:
        #         imgs.append(img)
        # return imgs

@MODULES.register_module()
class ScalePartyDiscriminator(nn.Module):
    """StyleGAN2 Discriminator.

    The architecture of this discriminator is proposed in StyleGAN2. More
    details can be found in: Analyzing and Improving the Image Quality of
    StyleGAN CVPR2020.

    Args:
        in_size (int): The input size of images.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4, channel_groups=1).
    """

    def __init__(self,
                 in_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 with_adaptive_pool=False,
                 pool_size=(2, 2),
                 input_channels=3):
        super().__init__()
        self.with_adaptive_pool = with_adaptive_pool
        self.pool_size = pool_size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(np.log2(in_size))
        in_channels = channels[in_size]
        convs = [ConvDownLayer(input_channels, channels[in_size], 1)]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]
            convs.append(ResBlock(in_channels, out_channel, blur_kernel))

            in_channels = out_channel

        self.convs = nn.Sequential(*convs)
        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)

        self.final_conv = ConvDownLayer(in_channels + 1, channels[4], 3)

        if self.with_adaptive_pool:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_size)
            linear_in_channels = channels[4] * pool_size[0] * pool_size[1]
        else:
            linear_in_channels = channels[4] * 4 * 4

        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                linear_in_channels,
                channels[4],
                act_cfg=dict(type='fused_bias')),
            EqualLinearActModule(channels[4], 1),
        )

    def forward(self, x, x_larger = None, input_is_fake=True):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        x = self.convs(x)

        x = self.mbstd_layer(x)
        x = self.final_conv(x)
        if self.with_adaptive_pool:
            x = self.adaptive_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)

        return x

@MODULES.register_module()
class DualScalePartyDiscriminator(ScalePartyDiscriminator):

    def __init__(self,
                 in_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 with_adaptive_pool=False,
                 pool_size=(2, 2),
                 input_channels=3,
                 channel_mix_prob = -1.,
                 crop_mix_prob = -1.,
                 sum_mix_prob = -1.,
                 two_disc = False,
                 scale_disc = False):
 
        super().__init__(
                in_size,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                mbstd_cfg=mbstd_cfg,
                with_adaptive_pool=with_adaptive_pool,
                pool_size=pool_size,
                input_channels=input_channels)

        self.crop_mix =  crop_mix_prob > 0
        self.channel_mix = channel_mix_prob > 0
        self.sum_mix = sum_mix_prob > 0
        self.crop_mix_prob = crop_mix_prob
        self.channel_mix_prob = channel_mix_prob 
        self.sum_mix_prob = sum_mix_prob 

        self.disc = super().forward
        if two_disc:
            self.disc_larger = ScalePartyDiscriminator(
                                in_size,
                                channel_multiplier=channel_multiplier,
                                blur_kernel=blur_kernel,
                                mbstd_cfg=mbstd_cfg,
                                with_adaptive_pool=with_adaptive_pool,
                                pool_size=pool_size,
                                input_channels=input_channels)
        else:
            self.disc_larger = self.disc

        self.scale_disc = None 
        if scale_disc:
            self.scale_disc = MultiScalePartyDiscriminator(
                                in_size=64,
                                channel_multiplier=channel_multiplier,
                                blur_kernel=blur_kernel,
                                mbstd_cfg=mbstd_cfg,
                                with_adaptive_pool=with_adaptive_pool,
                                pool_size=pool_size,
                                input_channels=input_channels)



    def forward(self, x, x_larger=None, input_is_fake=True):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.
            x_larger (torch.Tensor): Input image tensor but larger.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        if x_larger is None:
            return self.disc(x) 

        if self.scale_disc is not None:
            x_scale = self.scale_disc(x, x_larger)
        else:
            x_scale = None


        if input_is_fake and (self.crop_mix or self.channel_mix or self.sum_mix):
            x, x_larger = self.mix_images(x, x_larger)

        x= self.disc(x)
        x_larger= self.disc_larger(x_larger)

        if self.scale_disc is not None:
            return torch.cat([x, x_larger, x_scale], dim=0) # ConCat on batch dimension
        else:
            return torch.cat([x, x_larger], dim=0) # ConCat on batch dimension
    
    def mix_images(self, x, x_larger):
        large_x = F.interpolate(x, size=(x_larger.shape[-2], x_larger.shape[-1]),mode='bilinear')
        small_x_larger = F.interpolate(x_larger, size=(x.shape[-2], x.shape[-1]),mode='bilinear')
        if self.crop_mix and random.random() < self.crop_mix_prob:
            small_mask = self.get_mask(x[0], x.device)
            large_mask = self.get_mask(x_larger[0], x_larger.device)

            x, x_larger = \
                small_x_larger * small_mask + (1 - small_mask)*x, \
                large_x * large_mask + (1 - large_mask)*x_larger

        if self.channel_mix and random.random() < self.channel_mix_prob:
            channel_mask = self.get_channel_mask(x.device)
            x, x_larger = \
                small_x_larger * channel_mask + (1 - channel_mask)*x, \
                large_x * channel_mask + (1 - channel_mask)*x_larger

        if self.sum_mix and random.random() < self.sum_mix_prob:
            x, x_larger = \
                0.8*x + 0.2*small_x_larger, \
                0.2*x_larger + 0.8*large_x

        return x, x_larger

    def mix_images_detach(self, x, x_larger):
        small_x_larger = F.interpolate(x_larger, size=(x.shape[-2], x.shape[-1]),mode='bilinear').detach()
        if self.crop_mix and random.random() < self.crop_mix_prob:
            small_mask = self.get_mask(x[0], x.device)

            x = \
                small_x_larger * small_mask + (1 - small_mask)*x

        if self.channel_mix and random.random() < self.channel_mix_prob:
            channel_mask = self.get_channel_mask(x.device)
            x = \
                small_x_larger * channel_mask + (1 - channel_mask)*x

        if self.sum_mix:
            # x = x*0.8 + small_x_larger*0.2 
            channel_mask = 0
            small_mask = 0
            if random.random() < self.sum_mix_prob:
                channel_mask = self.get_channel_mask(x.device)
            if random.random() < self.sum_mix_prob:
                small_mask = self.get_mask(x[0], x.device)
            mask = 0.9 * channel_mask + 0.9 * small_mask  
            x = x - x.detach() * mask + small_x_larger* mask

        return x



    @staticmethod
    def get_mask(x, device):
        _, h, w = x.shape
        n = 1

        # r = torch.zeros(n,h,device=device)
        # start_r = random.randint(0, 3*h//4) 
        # r[:,start_r:min(start_r + h//4, h)] = 1

        # c = torch.zeros(n,w, device=device)
        # start_c = random.randint(0, 3*w//4) 
        # c[:,start_c:min(start_c + w//4, w)] = 1
 
        r = 1 - ((torch.rand(n,h+1,device=device).sort(1).indices <2).long().cumsum(1)-1).abs()[:,:h]
        c = 1 - ((torch.rand(n,w+1,device=device).sort(1).indices <2).long().cumsum(1)-1).abs()[:,:w]
        mask = (r.view(1,h,1) * c.view(1,1,w)).expand_as(x[:1])
        return mask

    @staticmethod
    def get_channel_mask(device):
        mask = torch.randint(high=2, size=(1,3,1,1),device=device)
        return mask

@MODULES.register_module()
class MultiScalePartyDiscriminator(ScalePartyDiscriminator):
    def __init__(self,
                 in_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 with_adaptive_pool=False,
                 pool_size=(2, 2),
                 input_channels=3):
 
        super().__init__(
                in_size,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                mbstd_cfg=mbstd_cfg,
                with_adaptive_pool=False,
                pool_size=pool_size,
                input_channels=2*input_channels)
        
        self.in_size = in_size

    def forward(self, x, x_larger, input_is_fake=True):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.
            x_larger (torch.Tensor): Input image tensor but larger.

        Returns:
            torch.Tensor: Predict score for the input image.
        """

        # x_larger = F.interpolate(x_larger, size=(x.shape[-2], x.shape[-1]), mode='bilinear')
        x = F.avg_pool2d(x,kernel_size = x.shape[-1]//self.in_size)
        x_larger = F.avg_pool2d(x_larger,kernel_size = x_larger.shape[-1]//self.in_size)
        x= torch.cat([x,x_larger], dim=1) # ConCat on channel dimension
        x= super().forward(x)

        return x

@MODULES.register_module()
class LatentScalePartyDiscriminator(ScalePartyDiscriminator):

    def __init__(self,
                 in_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 with_adaptive_pool=False,
                 pool_size=(2, 2),
                 input_channels=3):
        super().__init__(
                 in_size,
                 channel_multiplier=channel_multiplier,
                 blur_kernel=blur_kernel,
                 mbstd_cfg=mbstd_cfg,
                 with_adaptive_pool=with_adaptive_pool,
                 pool_size=pool_size,
                 input_channels=input_channels)
 
        if self.with_adaptive_pool:
            linear_in_channels = 512 * pool_size[0] * pool_size[1]
        else:
            linear_in_channels = 512 * 4 * 4

        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                linear_in_channels * 2,
                512,
                act_cfg=dict(type='fused_bias')),
            EqualLinearActModule(512, 1),
        )

    def forward(self, x, x_larger = None, input_is_fake=True):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        def get_disc_latent(x):
            x = self.convs(x)

            x = self.mbstd_layer(x)
            x = self.final_conv(x)
            if self.with_adaptive_pool:
                x = self.adaptive_pool(x)
            x = x.view(x.shape[0], -1)
            return x

        x = get_disc_latent(x)
        x_larger = get_disc_latent(x_larger)

        x = torch.cat([x,x_larger], dim=1) # ConCat on channel dimension
        x = self.final_linear(x)
        return x




