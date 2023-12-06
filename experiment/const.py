def pretrain_file(cfg):
    return f"{cfg.save_dir}{cfg.molecule_name}_{cfg.seed}_checkpoint_pretrain.ckpt"


def train_file(cfg):
    return f"{cfg.save_dir}{cfg.molecule_name}_{cfg.seed}.txt"


def trajectory_file(cfg, distance):
    return f"{cfg.save_dir}{cfg.molecule_name}_trajectory_{distance}.ckpt"


def image_file(cfg, errors):
    suffix = ""
    if errors is not None:
        suffix = "-detail"
    return f"{cfg.save_dir}result-{cfg.molecule_name}{suffix}.pdf"


def ground_state_file(cfg):
    return f"{cfg.save_dir}gs_{cfg.molecule_name}.txt"


def random_file(cfg, seed):
    return f'{cfg.save_dir}{cfg.molecule_name}_random_{seed}.txt'
