sampler = dict(
    type='InfoSampler',
)
encoder = dict(
    pos_encoder = dict(
        type='HashEncoder',
    ),
    dir_encoder = dict(
        type='SHEncoder',
    ),
)
model = dict(
    type='InfoNeRF',
    use_viewdirs=True,
)
fine_model = dict(
    type='InfoNeRF',
    use_viewdirs=True,
)
loss = dict(
    type='InfoLoss',
)
optim = dict(
    type='Adam',
    lr=5e-4,
    eps=1e-15,
    betas=(0.9,0.99),
)
ema = dict(
    type='EMA',
    decay=0.95,
)
# expdecay=dict(
#     type='ExpDecay',
#     decay_start=20_000,
#     decay_interval=10_000,
#     decay_base=0.33,
#     decay_end=None
# )
dataset_type = 'InfoNerfDataset'
dataset_dir = '/home/penghy/nerf_data/nerf_synthetic/lego'
dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=2048,
        mode='train',
    ),
    val=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=2048,
        mode='val',
        preload_shuffle=False,
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=2048,
        mode='test',
        preload_shuffle=False,
    ),
)

exp_name = "lego"
log_dir = "./logs"
tot_train_steps = 40000
# Background color, value range from 0 to 1
background_color = [0, 0, 0]
# Hash encoding function used in Instant-NGP
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 2048
n_training_steps = 16
# Expected number of sampling points per batch
# target_batch_size = 1<<19
# Set const_dt=True for higher performance
# Set const_dt=False for faster convergence
const_dt=True
# Use fp16 for faster training
fp16 = False
# Load pre-trained model
load_ckpt = False
# path of checkpoint file, None for default path
ckpt_path = None
# test output image with alpha
alpha_image= False
# InfoNerf Info.
near = 2.
far = 6.
use_viewdirs = True
perturb = 1.
multires = 10
multires_views = 4
raw_noise_std = 0.
white_bkgd = True
lrate_decay = 500
lindisp = False
pytest = False
no_coarse = False
lrate = 5e-4
# Entropy
N_samples = 64
N_importance = 128
entropy = True
N_entropy = 1024
# smoothing_lambda = 1.
entropy_ray_zvals_lambda = 0.001
precrop_iters = 500
precrop_frac = 0.5
no_batching = True
wandb = False
i_wandb = 10

half_res = False
fewshot = 4
train_scene = [26, 86, 2, 55]
# train_scene = [0,1,2,3]
entropy_type = "log2" # between log2 and 1-p
entropy_acc_threshold = 0.1