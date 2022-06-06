from .base_gan import BaseGAN
from .cyclegan import CycleGAN
from .mspie_stylegan2 import MSPIEStyleGAN2
from .pix2pix import Pix2Pix
from .progressive_growing_unconditional_gan import ProgressiveGrowingGAN
from .singan import PESinGAN, SinGAN
from .static_unconditional_gan import StaticUnconditionalGAN
from .scaleparty import ScaleParty

__all__ = [
    'BaseGAN', 'StaticUnconditionalGAN', 'ProgressiveGrowingGAN', 'SinGAN',
    'Pix2Pix', 'CycleGAN', 'MSPIEStyleGAN2', 'PESinGAN', "ScaleParty"
]
