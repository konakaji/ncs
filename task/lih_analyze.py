import lightning as L
import torch
from gqe.gptqe_model.transformer import Transformer
from task.lih import get_lih_configs

fabric = L.Fabric(accelerator="auto")
fabric.launch()
cfg = get_lih_configs()

model = Transformer(cfg, 'pretrain')
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
model, optimizer = fabric.setup(model, optimizer)
cp = fabric.load("checkpoints/gptqe/run_1016_18_12/checkpoint_1.0.ckpt")
model.load_state_dict(cp['model'])
indices = model.generate(ngates=3)

logits = model.generate_logits(indices)
