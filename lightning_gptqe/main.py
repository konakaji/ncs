import torch
# from transformers import get_cosine_schedule_with_warmup
from lightning_gptqe.experiments.hydrogen import cfg, find_ground_state_energy, find_computed_energy, plot_figure
from lightning_gptqe.models.transformer import Transformer
from pytorch_lightning.loggers import WandbLogger
import lightning as L

from datetime import datetime
import wandb

cfg.run_name = datetime.now().strftime("run_%m%d_%H_%M")
cfg.save_dir = f"checkpoints/{cfg.name}/{cfg.run_name}/"
logger = WandbLogger(
    project=cfg.name,
    name=cfg.run_name,
    log_model=True,
)
fabric = L.Fabric(accelerator="cpu", loggers=[logger])
fabric.seed_everything(cfg.seed)
fabric.launch()

computed_energies = []
distances = cfg.distances
for distance in distances:
    cost = find_ground_state_energy(distance, cfg)
    cfg.vocab_size = cost.vocab_size()
    model = Transformer(cfg, distance)
    model.set_cost(cost)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_iters)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.warmup_steps*10)
    model, optimizer = fabric.setup(model, optimizer)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total trainable params: {pytorch_total_params/1e6:.2f}M")
    model.train()
    for epoch in range(cfg.max_iters):
        optimizer.zero_grad()
        loss, log_values = model.train_step()
        fabric.log_dict(log_values)
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
        optimizer.step()
        # scheduler.step()
        model.temperature += 0.05
    model.set_cost(None)
    state = { "model": model, "optimizer": optimizer, "hparams": model.hparams }
    fabric.save(cfg.save_dir+f"checkpoint_{distance}.ckpt", state)
    computed_energies.append(find_computed_energy(cost, model))
    
plt, impath = plot_figure(cfg, distances, computed_energies)
fabric.log('result', wandb.Image(plt))
