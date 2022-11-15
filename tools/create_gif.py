# %%
from random import randint, random, randrange
import torch 
import torch.nn.functional as F

from mmgen.core.evaluation.metrics import ms_ssim
from PIL import Image
import torch.nn.functional as F
import tqdm



from mmgen.apis import init_model, sample_uncoditional_model  # isort:skip  # noqa
from mmgen.models.architectures import positional_encoding

import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

import torch
from scipy import ndimage
# from skimage import warp, AffineTransform
import warnings
import os


# %%
def make_injected_noise(chosen_scale=0, device='cuda'):
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


def prepare_img(img, img_to_get_size=None):
    minibatch = img

    if img_to_get_size is not None:
        minibatch = F.interpolate(minibatch, size=img_to_get_size.shape[-2:], mode='bilinear')
    minibatch = ((minibatch + 1) / 2)
    minibatch = minibatch.clamp_(0, 1)
    minibatch = minibatch[:, [2, 1, 0], :, :]
    minibatch = minibatch.cpu().data.numpy().transpose((0, 2, 3, 1))
    minibatch = (minibatch * 255).astype('uint8')

    return minibatch


# print(noise_sizes)
run_name = "0_ScaleParty_Full"
epoch="latest"

config = f"configs/scaleparty/{run_name}.py"
ckpt = f"checkpoints/{run_name}/ckpt/{epoch}.pth"
device= 'cuda' if torch.cuda.is_available() else 'cpu'

model = init_model(
    config, checkpoint=ckpt, device=device)
get_pos = model.generator.head_pos_enc.make_grid2d
model.eval()

out_size = model.generator.out_size
enc_size = model.generator.head_pos_size[0]


batch_size = 1

# %% is_style =False
latent = model.generator.get_noise(num_batches=batch_size)
injected_noise = make_injected_noise(chosen_scale=2, device=device)


# %%
num_outputs = 1
for i in range(num_outputs):
    iters = 1 
    scale=8
    sub_step_size = 2
    is_style = False
    use_noise = True 
    intrp_noise = True 

    latent = model.generator.get_noise(num_batches=batch_size)
    injected_noise = make_injected_noise(chosen_scale=scale-4, device=device)


    # injected_noise = [ torch.zeros_like(x, device=device)
                    #  for x in injected_noise]

    scale_range = range(100,151, 1)
    # scale_range = range(100, 29, -5)
    shift_range = [0] # range(-128,129,10)
    ssim_res = np.zeros((len(scale_range),len(scale_range)))
    results = []

    for scale_len in scale_range:
        shift_h = 0
        for shift_w in shift_range:
            lenght = scale_len/100 

            positions = get_pos(
                                scale, 
                                scale,
                                num_batches=batch_size,
                                # start_h= 0,
                                start_h= 0.5 - lenght/2 + shift_h/(lenght/256),
                                # start_w= 0,
                                start_w= 0.5 - lenght/2 + shift_w*(lenght/256),
                                width=  lenght,
                                height= lenght).to(device)

            pos_ref = get_pos(
                                scale, 
                                scale,
                                num_batches=batch_size,
                                start_h= 0,
                                # start_h= 0.,
                                start_w= 0,
                                # start_w= 0.,
                                width=  1.,
                                height= 1.).to(device)


            if use_noise:
                if intrp_noise:
                    new_pos = positions * pos_ref.max()/ positions.max()
                    new_injected_noise = []

                    new_pos = F.avg_pool2d(new_pos, kernel_size=[3,3], stride=1)
                    pos_ref = F.avg_pool2d(pos_ref, kernel_size=[3,3], stride=1)

                    new_injected_noise.append(F.grid_sample(injected_noise[0], new_pos.permute(0,2,3,1)))
                    # new_injected_noise.append(F.grid_sample(whole_noise, new_pos.permute(0,2,3,1)))
                    for noise_map_1, noise_map_2 in zip(injected_noise[1::2], injected_noise[2::2]):
                        pos_ref = F.interpolate(pos_ref, scale_factor=2, mode="bilinear")
                        pos_ref = F.avg_pool2d(pos_ref, kernel_size=[3,3], stride=1)

                        new_pos = F.interpolate(new_pos, scale_factor=2, mode="bilinear")
                        new_pos = F.avg_pool2d(new_pos, kernel_size=[3,3], stride=1)

                        new_pos = new_pos * pos_ref.max()/ new_pos.max()
                        new_injected_noise.append(F.grid_sample(noise_map_1, new_pos.permute(0,2,3,1),mode='nearest'))

                        pos_ref = F.avg_pool2d(pos_ref, kernel_size=[3,3], stride=1)
                        new_pos = F.avg_pool2d(new_pos, kernel_size=[3,3], stride=1)

                        new_pos = new_pos * pos_ref.max()/ new_pos.max()
                        new_injected_noise.append(F.grid_sample(noise_map_2, new_pos.permute(0,2,3,1),mode='nearest'))
                    new_injected_noise = new_injected_noise
                else:
                    new_injected_noise = injected_noise 
            else:
                new_injected_noise=None


            minibatch = model.generator_ema(
                latent, injected_noise=new_injected_noise,
                num_batches=batch_size, chosen_scale=scale-4,  input_is_latent = is_style,
                positional_enc= positions, truncation=0.6)

            

            full_size = int(minibatch.shape[-1]/lenght )
            start = (minibatch.shape[-1] - full_size)//2
            

            img = minibatch
            results.append(prepare_img(img)[0]) 
            plt.imshow(results[-1])

            plt.show()


    import imageio 
    if not use_noise:
        extra = "random"
    elif intrp_noise:
        extra = "intrp"
    else: 
        extra = "constant"

    imageio.mimsave(f"{run_name}_{extra}_{i}.gif", results)


