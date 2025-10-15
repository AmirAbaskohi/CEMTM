import os
import torch
from torch.utils.data import DataLoader
from losses.losses import ReconstructionLoss, EntropyLoss, KLDivergenceLoss
from trainer.utils import set_seed, build_optimizer, build_scheduler, save_checkpoint
from data.dataset import get_dataset
from model.cemtm import CEMTM
from tqdm import tqdm


def train_cemtm(config):
    # === Set random seed ===
    set_seed(config["seed"])

    # === Load dataset with optional lazy loading to avoid memory overflow ===
    batch_size = config["training"]["batch_size"]
    lazy_loading = config["data"].get("lazy_loading", True)  # Default to True for memory efficiency
    
    dataset = get_dataset(
        config["data"]["name"], 
        config["data"]["dataset_path"],
        lazy=lazy_loading,
        batch_size=batch_size
    )
    
    # Note: IterableDataset doesn't support shuffling in DataLoader
    # For lazy loading, data is streamed in order; for shuffling, use eager loading
    shuffle = not lazy_loading
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # === Setup model ===
    model = CEMTM(
        vlm_model_name=config["vlm2vec"]["model_name"],
        input_dim=config["model"]["d_model"],
        num_topics=config["model"]["n_topics"],
        transformer_layers=config["model"]["transformer_layers"],
        transformer_heads=config["model"]["transformer_heads"],
        dropout=config["model"]["dropout"],
        freeze_vlm=config["vlm2vec"]["freeze"],
    ).to(config["training"]["device"])

    # === Setup optimizer, scheduler ===
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, num_training_steps=len(dataloader) * config["training"]["num_epochs"])

    # === Loss functions ===
    recon_loss_fn = ReconstructionLoss()
    entropy_loss_fn = EntropyLoss()
    kl_loss_fn = KLDivergenceLoss()

    # === Training loop ===
    model.train()
    for epoch in range(config["training"]["num_epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            text = batch["text"]
            images = batch["images"]

            optimizer.zero_grad()

            # Forward pass
            output = model(text[0], images[0])  # batch size = 1 for now

            # Losses
            L_rec = recon_loss_fn(output["e_d_prime"], output["e_d"])
            L_ent = entropy_loss_fn(output["beta"])
            L_kl = kl_loss_fn(output["mu"], output["logvar"])

            # Combined loss
            loss = (
                L_rec
                + config["loss"]["lambda_entropy"] * L_ent
                + config["loss"]["lambda_kl"] * L_kl
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({
                "L_rec": L_rec.item(),
                "L_ent": L_ent.item(),
                "L_kl": L_kl.item(),
                "Loss": loss.item()
            })

        # Save model
        if (epoch + 1) % config["training"]["save_every"] == 0:
            save_path = os.path.join(config["output"]["save_dir"], f"cemtm_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, save_path)
