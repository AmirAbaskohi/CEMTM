import os
import torch
import random
import numpy as np
from transformers import get_scheduler


def set_seed(seed: int = 42):
    """
    Ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, config):
    """
    Build optimizer (default: AdamW)
    """
    optim_config = config["optimizer"]
    lr = float(optim_config["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=optim_config.get("weight_decay", 0.01),
    )
    return optimizer


def build_scheduler(optimizer, config, num_training_steps):
    """
    Scheduler using Hugging Face's get_scheduler
    """
    sched_config = config["scheduler"]
    scheduler = get_scheduler(
        name=sched_config["name"],
        optimizer=optimizer,
        num_warmup_steps=sched_config.get("warmup_steps", 0),
        num_training_steps=num_training_steps,
    )
    return scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """
    Save training checkpoint
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(model, optimizer, scheduler, load_path, device="cuda"):
    """
    Load training checkpoint
    """
    state = torch.load(load_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"]
