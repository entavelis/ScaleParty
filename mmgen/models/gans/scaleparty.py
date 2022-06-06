import logging
from functools import partial

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import _find_tensors

import random 

from ..builder import MODELS
from ..common import set_requires_grad
from .static_unconditional_gan import StaticUnconditionalGAN

from mmgen.models.builder import MODELS, build_module

@MODELS.register_module()
class ScaleParty(StaticUnconditionalGAN):
    """MS-PIE StyleGAN2.

    In this GAN, we adopt the MS-PIE training schedule so that multi-scale
    images can be generated with a single generator. Details can be found in:
    Positional Encoding as Spatial Inductive Bias in GANs, CVPR2021.

    Args:
        generator (dict): Config fo.r generator.
        discriminator (dict): Config for discriminator.
        gan_loss (dict): Config for generative adversarial loss.
        disc_auxiliary_loss (dict): Config for auxiliary loss to
            discriminator.
        gen_auxiliary_loss (dict | None, optional): Config for auxiliary loss
            to generator. Defaults to None.
        train_cfg (dict | None, optional): Config for training schedule.
            Defaults to None.
        test_cfg (dict | None, optional): Config for testing schedule. Defaults
            to None.
    """

    def __init__(self,
                generator,
                discriminator,
                gan_loss=None,
                disc_auxiliary_loss=None,
                gen_auxiliary_loss=None,
                train_cfg=None,
                test_cfg=None,
                scale_loss=None,
                distributed=False):
        super().__init__(
                generator,
                discriminator,
                gan_loss,
                disc_auxiliary_loss=disc_auxiliary_loss,
                gen_auxiliary_loss=gen_auxiliary_loss,
                train_cfg=train_cfg,
                test_cfg=test_cfg)

        if (scale_loss is not None) and self.multiscale: 
            self.scale_loss = build_module(scale_loss) 
        else:
            self.scale_loss = None

        self.min_resolution = generator['head_pos_encoding']['min_resolution']
        self.max_resolution = generator['head_pos_encoding']['max_resolution']

        # if distributed:
            # self.generator = self.generator.module
            # self.discriminator = self.discriminator.module
    def _parse_train_cfg(self):
        super(ScaleParty, self)._parse_train_cfg()

        # set the number of upsampling blocks. This value will be used to
        # calculate the current result size according to the size of the input
        # feature map, e.g., positional encoding map
        self.num_upblocks = self.train_cfg.get('num_upblocks', 6)

        # multiple input scales (a list of int) that will be added to the
        # original starting scale.
        self.multiscale = self.train_cfg.get('multiscale')
        self.multiscale_chance = self.train_cfg.get('multiscale_chance')
        print(self.multiscale_chance)
        # self.multiscale_error = self.train_cfg.get('multiscale_error')
        # self.l_ms = self.train_cfg.get('l_ms')
        self.extra_scale = self.train_cfg.get('extra_scale')
        # self.full_resolution = self.train_cfg.get('full_resolution')

        self.multi_input_scales = self.train_cfg.get('multi_input_scales')
        self.multi_scale_probability = self.train_cfg.get(
            'multi_scale_probability')


    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
        """Train step function.

        This function implements the standard training iteration for
        asynchronous adversarial training. Namely, in each iteration, we first
        update discriminator and then compute loss for generator with the newly
        updated discriminator.

        As for distributed training, we use the ``reducer`` from ddp to
        synchronize the necessary params in current computational graph.

        Args:
            data_batch (dict): Input data from dataloader.
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
        # get data from data_batch
        real_imgs = data_batch['real_img']
        real_imgs_x2 = None

        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = real_imgs.shape[0]

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        if dist.is_initialized():
            # randomly sample a scale for current training iteration
            if self.multiscale or random.random() < self.multiscale_chance:
                chosen_scale = 0
                chosen_scale = torch.tensor(chosen_scale, dtype=torch.int).cuda()
                multiscale = torch.tensor(True, dtype=torch.bool).cuda()
            else:
                chosen_scale = np.random.choice(self.multi_input_scales, 1,
                                                self.multi_scale_probability)[0]

                chosen_scale = torch.tensor(chosen_scale, dtype=torch.int).cuda()
                multiscale = torch.tensor(False, dtype=torch.bool).cuda()

            dist.broadcast(chosen_scale, 0)
            dist.broadcast(multiscale, 0)
            chosen_scale = int(chosen_scale.item())

        else:
            mmcv.print_log(
                'Distributed training has not been initialized. Degrade to '
                'the standard stylegan2',
                logger='mmgen',
                level=logging.WARN)
            chosen_scale = 0
            multiscale=False

        full_image_resolution = min(real_imgs.shape[-2:])
        curr_relative_length = random.uniform(self.min_resolution, self.max_resolution)
        curr_length = int(curr_relative_length*full_image_resolution) 

        ## Crop image
        if curr_length > full_image_resolution:
            real_imgs = F.interpolate(
                real_imgs,
                size=(curr_length, curr_length),
                mode='bilinear',
                align_corners=False)
            
            curr_length = full_image_resolution



        start_h = random.randint(0, real_imgs.shape[-1] - curr_length)
        start_w = random.randint(0, real_imgs.shape[-2]  - curr_length)
        real_imgs = real_imgs[:,:,start_h:start_h+curr_length,start_w:start_w+curr_length]

       
        # adjust the shape of images
        if multiscale:
            large_curr_size = (4+self.extra_scale + chosen_scale) * (2**self.num_upblocks)
            if real_imgs.shape[-2:] != (large_curr_size, large_curr_size):
                real_imgs_x2 = F.interpolate(
                    real_imgs,
                    size=(large_curr_size, large_curr_size),
                    mode='bilinear',
                    align_corners=False)
                    # mode='bilinear',
                    # align_corners=True)

            large_positional_enc = self.generator.head_pos_enc.make_grid2d(
                                                                        n_height=4 + self.extra_scale + chosen_scale,
                                                                        n_width=4 + self.extra_scale + chosen_scale,
                                                                        start_h=start_h/full_image_resolution,
                                                                        start_w=start_w/full_image_resolution,
                                                                        width=curr_relative_length,
                                                                        height=curr_relative_length,
                                                                        num_batches=batch_size
                                                                        )
                                                                        




        small_curr_size = (4 + chosen_scale) * (2**self.num_upblocks)
        if real_imgs.shape[-2:] != (small_curr_size, small_curr_size):
            real_imgs = F.interpolate(
                real_imgs,
                size=(small_curr_size, small_curr_size),
                mode='bilinear',
                align_corners=False)
                # mode='bilinear',
                # align_corners=True)

        small_positional_enc = self.generator.head_pos_enc.make_grid2d(
                                                                    n_height=4 + chosen_scale,
                                                                    n_width=4 + chosen_scale,
                                                                    start_h=start_h/full_image_resolution,
                                                                    start_w=start_w/full_image_resolution,
                                                                    width=curr_relative_length,
                                                                    height=curr_relative_length,
                                                                    num_batches=batch_size
                                                                    )
                                                                    
        # disc training
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()
        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            noise = self.generator.get_noise(batch_size)
            fake_imgs = self.generator(
                noise, num_batches=batch_size, chosen_scale=chosen_scale,
                positional_enc = small_positional_enc)

            if multiscale:
                fake_imgs_x2 = self.generator(
                    noise, num_batches=batch_size, chosen_scale=self.extra_scale + chosen_scale,
                    positional_enc=large_positional_enc)
            else:
                fake_imgs_x2 = None


        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs, fake_imgs_x2, input_is_fake=True)
        disc_pred_real = self.discriminator(real_imgs, real_imgs_x2, input_is_fake=False)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            real_imgs_x2=real_imgs_x2,
            iteration=curr_iter,
            batch_size=batch_size,
            gen_partial=partial(self.generator, chosen_scale=self.extra_scale + chosen_scale))

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))
        loss_disc.backward()
        optimizer['discriminator'].step()

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
            log_vars_disc['curr_size'] = curr_relative_length 
            outputs = dict(
                log_vars=log_vars_disc,
                num_samples=batch_size,
                results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        # generator training
        set_requires_grad(self.discriminator, False)
        optimizer['generator'].zero_grad()

        # TODO: add noise sampler to customize noise sampling
        noise = self.generator.get_noise(batch_size)
        fake_imgs = self.generator(
            noise, num_batches=batch_size, chosen_scale=chosen_scale,
            positional_enc= small_positional_enc)

        if multiscale:
            fake_imgs_x2 = self.generator(
                noise, num_batches=batch_size, chosen_scale=self.extra_scale + chosen_scale,
                positional_enc = large_positional_enc)
        else:
            fake_imgs_x2 = None

        disc_pred_fake_g = self.discriminator(fake_imgs, fake_imgs_x2, input_is_fake=True)

        if multiscale and self.scale_loss:
            fake_imgs_x2= F.interpolate(fake_imgs_x2,
                                        size=(fake_imgs.shape[-2],fake_imgs.shape[-1]),
                                        mode='bilinear')

        if multiscale:
            gen_partial = random.choice([
                        partial(self.generator, positional_enc=small_positional_enc, chosen_scale=chosen_scale),
                        partial(self.generator, positional_enc=large_positional_enc, chosen_scale=chosen_scale + self.extra_scale),
                        ])
        else:
            gen_partial = partial(self.generator, positional_enc=small_positional_enc, chosen_scale=chosen_scale),

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            fake_imgs_x2=fake_imgs_x2,
            disc_pred_fake_g=disc_pred_fake_g,
            iteration=curr_iter,
            batch_size=batch_size,
            gen_partial=gen_partial,
            )

        loss_gen, log_vars_g = self._get_gen_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

        loss_gen.backward()
        optimizer['generator'].step()

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)
        log_vars['curr_size'] = curr_relative_length 

        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    def _get_gen_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake_g'] = self.gan_loss(
            outputs_dict['disc_pred_fake_g'],
            target_is_real=True,
            is_disc=False)

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                if loss_ is None:
                    continue

                # mmcv.print_log(f'get loss for {loss_module.name()}')
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_

        # gen auxiliary loss
        if self.scale_loss:
            loss_ = self.scale_loss(outputs_dict)

            # mmcv.print_log(f'get loss for {loss_module.name()}')
            # the `loss_name()` function return name as 'loss_xxx'
            if self.scale_loss.loss_name() in losses_dict:
                losses_dict[self.scale_loss.loss_name(
                )] = losses_dict[self.scale_loss.loss_name()] + loss_
            else:
                losses_dict[self.scale_loss.loss_name()] = loss_


        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

