from ml_collections import ConfigDict


def get_default_pretrain_configs():
    cfg = ConfigDict()
    cfg.name = "gptqe"
    cfg.temperature = 0
    cfg.grad_norm_clip = 1.0
    cfg.nqubit = 4
    cfg.ngates = 20
    cfg.seed = 3047
    cfg.lr = 5e-7
    cfg.energy_offset = 0.0
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.max_iters = 5
    cfg.nshot = 0
    cfg.check_points = {}
    cfg.verbose = False
    return cfg


def get_default_configs():
    cfg = ConfigDict()
    cfg.tool = "qulacs"
    cfg.verbose = False
    cfg.save_data = False
    cfg.print_exact = True
    cfg.name = "gptqe"
    cfg.backward_frequency = 1
    cfg.warmup_steps = 100
    cfg.num_samples = 5  # akin to batch size
    cfg.max_iters = 100
    cfg.nqubit = 4
    cfg.ngates = 20
    cfg.nshot = 0
    cfg.seed = 3047
    cfg.transformation = 'jordan-wigner'
    cfg.distances = [0.5, 0.6, 0.7, 0.7414, 0.8, 0.9, 1.0, 1.5, 2.0]  # choices of the distance between two atoms
    cfg.time_pool = [0.003125, -0.003125, 0.00625, -0.00625, 0.0125, -0.0125, 0.025, -0.025, 0.05, -0.05, 0.1, -0.1]
    cfg.time_factor = 1.0
    cfg.is_bravyi = cfg.transformation == 'bravyi-kitaev'
    cfg.lr = 5e-7
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 5.0
    cfg.del_temperature = 0.05
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.check_points = {}
    cfg.dry = False
    cfg.small = False
    cfg.cache = True
    cfg.save_dir = "../output/"
    return cfg
