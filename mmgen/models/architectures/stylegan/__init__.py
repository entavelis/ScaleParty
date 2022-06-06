from .generator_discriminator_v1 import (StyleGAN1Discriminator,
                                         StyleGANv1Generator)
from .generator_discriminator_v2 import (StyleGAN2Discriminator,
                                         StyleGANv2Generator)
from .mspie import MSStyleGAN2Discriminator, MSStyleGANv2Generator
from .scaleparty import ScalePartyDiscriminator, ScalePartyGenerator

__all__ = [
    'StyleGAN2Discriminator', 'StyleGANv2Generator', 'StyleGANv1Generator',
    'StyleGAN1Discriminator', 'MSStyleGAN2Discriminator',
    'MSStyleGANv2Generator', 'ScalePartyDiscriminator', 'ScalePartyGenerator' 
]
