import numpy as np
import torch
import torch.nn as nn

from random import randint, uniform

from mmgen.models.builder import MODULES


@MODULES.register_module('SPE')
@MODULES.register_module('SPE2d')
class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).

    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa

    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.

    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].

        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)

        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]

        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)


@MODULES.register_module('CSG2d')
@MODULES.register_module('CSG')
@MODULES.register_module()
class CatersianGrid(nn.Module):
    """Catersian Grid for 2d tensor.

    The Catersian Grid is a common-used positional encoding in deep learning.
    In this implementation, we follow the convention of ``grid_sample`` in
    PyTorch. In other words, ``[-1, -1]`` denotes the left-top corner while
    ``[1, 1]`` denotes the right-botton corner.
    """

    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    def make_grid2d(self, height, width, num_batches=1, requires_grad=False):
        h, w = height, width
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = torch.stack((grid_x, grid_y), 0)
        grid.requires_grad = requires_grad

        grid = torch.unsqueeze(grid, 0)
        grid = grid.repeat(num_batches, 1, 1, 1)

        return grid

    def make_grid2d_like(self, x, requires_grad=False):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)

@MODULES.register_module('ScaleParty')
@MODULES.register_module('SP')
class ScalePartyPE(nn.Module):
    def __init__(self,
                 min_resolution=256,
                 max_resolution=512,
                 pad=6,
                 displacement = False,
                 disp_level = 10,
                 clamp = False,
                 ):
        super().__init__()
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.pad = pad 
        self.clamp = clamp 
        assert pad % 2 == 0


    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    # def make_grid2d(self, n_height, n_width, num_batches=1, start_h=None, start_w=None, width=None, height=None, requires_grad=False):
    def make_grid2d(self, n_height, n_width, num_batches=1, start_h=0, start_w=0, width=1.0, height=1.0, requires_grad=False):

        max_resolution =  self.max_resolution
        min_resolution = self.min_resolution
        half_pad = self.pad // 2

        if not width and not height:
            width = height = uniform(min_resolution, max_resolution)

        pe = torch.zeros(2, n_height + self.pad, n_width + self.pad)

        if start_h is None:
            start_h = uniform(-1, 1 - 2*height)
        else:
            start_h = start_h*2 - 1
        if start_w is None:
            start_w = uniform(-1, 1 - 2*width) 
        else:
            start_w = start_w*2 - 1

        step_w = 2 * width /  n_width
        step_h = 2 * height / n_height

        start_w += step_w/2
        start_h += step_h/2
        # start_w = (2 * start_w) / max_resolution - 1 + (step_w / 2)  
        # start_h = (2 * start_h) / max_resolution - 1 + (step_h / 2)  

        pos_w = torch.arange(start_w - (half_pad * step_w), start_w + (n_width + half_pad) * step_w, step_w)[:(n_width+self.pad)].unsqueeze(1)
        pos_h = torch.arange(start_h - (half_pad * step_h), start_h + (n_height + half_pad) * step_h, step_h)[:(n_height+self.pad)].unsqueeze(1)

        if self.clamp:
            pos_w = pos_w.clamp_(-1, 1)
            pos_h = pos_h.clamp_(-1, 1)

        pe[0] = pos_w.transpose(0,1).repeat(1, n_width+self.pad, 1)
        pe[1] = pos_h.repeat(1, 1, n_height+self.pad)
        

        pe = pe.repeat(num_batches, 1, 1, 1)
        return pe

 
        # return grid

    def make_grid2d_like(self, x, requires_grad=False):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)
@MODULES.register_module('ScalePartySPE')
@MODULES.register_module('SPSPE')
class ScalePartySPEPE(ScalePartyPE):
    def __init__(self, min_resolution=256, max_resolution=512, pad=6):
        super().__init__(min_resolution=min_resolution, max_resolution=max_resolution, pad=pad)

        self.embedding_dim = 512
 

    def make_grid2d(self, n_height, n_width, num_batches=1, start_h=0, start_w=0, width=1.0, height=1.0, requires_grad=False):

        max_resolution =  self.max_resolution
        min_resolution = self.min_resolution
        half_pad = self.pad // 2

        if not width and not height:
            width = height = uniform(min_resolution, max_resolution)

        pe = torch.zeros(self.embedding_dim, n_width + self.pad, n_height + self.pad)

        if start_h is None:
            start_h = uniform(-1, 1 - 2*height)
        else:
            start_h = start_h*2 - 1
        if start_w is None:
            start_w = uniform(-1, 1 - 2*width) 
        else:
            start_w = start_w*2 - 1

        step_w = 2 * width /  n_width
        step_h = 2 * height / n_height

        start_w += step_w/2
        start_h += step_h/2
        # start_w = (2 * start_w) / max_resolution - 1 + (step_w / 2)  
        # start_h = (2 * start_h) / max_resolution - 1 + (step_h / 2)  

        pos_w = torch.arange(start_w - (half_pad * step_w), start_w + (n_width + half_pad) * step_w, step_w)[:(n_width+self.pad)].unsqueeze(1)
        pos_h = torch.arange(start_h - (half_pad * step_h), start_h + (n_height + half_pad) * step_h, step_h)[:(n_height+self.pad)].unsqueeze(1)

        half_dim = self.embedding_dim // 4
        # emb = np.log(10000) / (half_dim - 1)

        # compute exp(-log10000 / d * i)
        # emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(half_dim, dtype=torch.float) + 1

        emb_w = pos_w * emb.unsqueeze(0)
        emb_w = torch.cat([torch.sin(emb_w), torch.cos(emb_w)],
                        dim=1)

        emb_h = pos_h * emb.unsqueeze(0)
        emb_h = torch.cat([torch.sin(emb_h), torch.cos(emb_h)],
                        dim=1)
 
        pe[:2*half_dim] = emb_w.transpose(0,1).unsqueeze(1).repeat(1, n_width+self.pad, 1)
        pe[2*half_dim:] = emb_h.transpose(0,1).unsqueeze(2).repeat(1, 1, n_height+self.pad)
        

        pe = pe.repeat(num_batches, 1, 1, 1)
        return pe
