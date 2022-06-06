_base_ = [
    '../_base_/datasets/ffhq_flip.py', '../_base_/models/stylegan2_base.py',
    '../_base_/default_runtime.py'
]

# use_ddp_wrapper = True
# find_unused_parameters = False

# runner = dict(
#     type='DynamicIterBasedRunner',
#     is_dynamic_ddp=False,
#     pass_training_status=True)


model = dict(
    type='ScaleParty',
    distributed=False,
    generator=dict(
        type='ScalePartyGenerator',
        channel_multiplier=1, # NOTE faster mode
        head_pos_encoding=dict(
            type='ScaleParty',
            min_resolution=0.5,
            max_resolution=1.0,
            pad=6),
        deconv2conv=True,
        up_after_conv=False,
        no_pad = True,
        interp_pad=False,
        up_config=dict(scale_factor=2, mode='bilinear', align_corners=False),
        out_size=128, # NOTE faster mode
        use_noise=False,
        ),
        
    discriminator=dict(
        type='DualScalePartyDiscriminator', in_size=128, with_adaptive_pool=True),

    # scale_loss=dict(
    #     type='L1Loss',
    #     loss_weight=0.1,
    #     reduction='mean',
    #     data_info=dict(
    #         pred='fake_imgs_x2',
    #         target='fake_imgs'
    #     )
    # )
)

train_cfg = dict(
    num_upblocks=5, ## NOTE faster mode
    multiscale=True,
    multiscale_chance = 0,
    extra_scale=2,
    multiscale_error='l1',
    l_ms='0.1',
    full_resolution=256,
    multi_input_scales=[0],
    multi_scale_probability=[1]
    )

data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root='./style_data')),
    # val='./style_data'
    )

ema_half_life = 10.
custom_hooks = [
    dict(
        type='VisualizeWandb',
        output_dir='training_samples',
        interval=1000),
    #  dict(
    #     type='VisualizeUnconditionalSamples',
    #     output_dir='training_samples',
    #     interval=5000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),
        priority='VERY_HIGH')
]

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=40)
lr_config = None

cudnn_benchmark = False
# total_iters = 1100002
total_iters = 625000

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl='ffhq_256.pkl',
        bgr2rgb=True),
    pr10k3=dict(type='PR', num_images=10000, k=3))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='MultiscaleMetrics',
        num_images=50000,
        inception_pkl='ffhq-256-50k-rgb.pkl',
        bgr2rgb=True),
    sample_kwargs=dict(
        sample_model='ema',
        double_eval=True,
        )
    )

wandb_config = {"model":model,
                "data": data,
                "train_cfg": train_cfg}
run_name = "scaleparty"
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook', 
            init_kwargs= dict(
                project='ScaleParty',
                name=run_name,
                config=wandb_config
            )
        )
    ])

