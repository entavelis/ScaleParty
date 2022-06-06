_base_ = [
    'lsun_base.py'
]

d_reg_interval = 16

model = dict(
    generator=dict(
        type='ScalePartyGenerator',
        channel_multiplier=1, # NOTE faster mode
        head_pos_encoding=dict(
            type='ScaleParty',
            min_resolution=0.6,
            max_resolution=1.1,
            pad=6),
        deconv2conv=True,
        up_after_conv=False,
        no_pad = True,
        interp_pad=False,
        up_config=dict(scale_factor=2, mode='bilinear', align_corners=False),
        out_size=128, # NOTE faster mode
        use_noise=True,
        ),
        
    disc_auxiliary_loss=dict(
        type='R1GradientPenalty',
        loss_weight=10. / 2. * d_reg_interval,
        interval=d_reg_interval,
        norm_mode='HWC',
        data_info=dict(real_data='real_imgs', real_data_x2='real_imgs_x2', discriminator='disc')),
    discriminator=dict(
        type='DualScalePartyDiscriminator', 
        in_size=128,
        with_adaptive_pool=True,
        crop_mix_prob=1.0,
        channel_mix_prob=1.0,
        )
)

train_cfg = dict(
    num_upblocks=5, ## NOTE faster mode
    multiscale=False,
    multiscale_chance=0.20,
    extra_scale=2,
    multiscale_error='l1',
    l_ms='0.1',
    full_resolution=256,
    multi_input_scales=[0,2],
    multi_scale_probability=[0.5,0.5]
    )


data = dict(
    samples_per_gpu=8,
    train=dict(dataset=dict(imgs_root='./style_data')))

run_name ='lsun_bedroom_part_multiscalechance_mix02'
wandb_config = {"model":model,
                "data": data,
                "train_cfg": train_cfg}

log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook', 
            init_kwargs= dict(
                project='mmScaleParty',
                name=run_name,
                config=wandb_config
            )
        )
    ])



