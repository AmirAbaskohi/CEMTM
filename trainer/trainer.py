import os
import torch
import json
from torch.utils.data import DataLoader
from losses.losses import ReconstructionLoss, EntropyLoss, KLDivergenceLoss
from trainer.utils import set_seed, build_optimizer, build_scheduler, save_checkpoint
from trainer.collate import custom_collate_fn
from data.dataset import get_dataset
from data.preprocessing import clean_text, build_vocab
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
    
    # === Build vocabulary from corpus ===
    print("Building vocabulary from corpus...")
    vocab_path = os.path.join(config["output"]["save_dir"], "vocabulary.json")
    
    if os.path.exists(vocab_path):
        print(f"Loading existing vocabulary from {vocab_path}")
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
    else:
        # Collect all text data for vocabulary building
        print("Collecting corpus for vocabulary building...")
        corpus = []
        if lazy_loading:
            # Need to iterate once to build vocab
            temp_dataset = get_dataset(
                config["data"]["name"],
                config["data"]["dataset_path"],
                lazy=True,
                batch_size=batch_size
            )
            for sample in tqdm(temp_dataset, desc="Collecting texts"):
                cleaned = clean_text(sample["text"])
                corpus.append(cleaned)
        else:
            for sample in tqdm(dataset, desc="Collecting texts"):
                cleaned = clean_text(sample["text"])
                corpus.append(cleaned)
        
        # Build vocabulary
        vocab = build_vocab(
            corpus, 
            max_vocab_size=config["data"].get("vocab_size", 30000),
            min_freq=config["data"].get("min_word_freq", 5)
        )
        
        # Save vocabulary
        os.makedirs(config["output"]["save_dir"], exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        print(f"Vocabulary saved to {vocab_path}")
    
    vocab_set = set(vocab)  # Convert to set for faster lookup
    
    # Note: IterableDataset doesn't support shuffling in DataLoader
    # For lazy loading, data is streamed in order; for shuffling, use eager loading
    shuffle = not lazy_loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,  # Use custom collate function for PIL images
        num_workers=0  # Keep 0 for multimodal data; increase cautiously
    )

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
            texts = batch["text"]  # List of strings
            images_list = batch["images"]  # List of image lists
            
            # Process batch (support variable batch sizes)
            batch_loss = 0.0
            batch_l_rec = 0.0
            batch_l_ent = 0.0
            batch_l_kl = 0.0
            
            for text, images in zip(texts, images_list):
                optimizer.zero_grad()

                # Forward pass
                output = model(text, images)

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

                # Accumulate for logging
                batch_loss += loss.item()
                batch_l_rec += L_rec.item()
                batch_l_ent += L_ent.item()
                batch_l_kl += L_kl.item()
            
            # Average over batch samples
            num_samples = len(texts)
            pbar.set_postfix({
                "L_rec": batch_l_rec / num_samples,
                "L_ent": batch_l_ent / num_samples,
                "L_kl": batch_l_kl / num_samples,
                "Loss": batch_loss / num_samples
            })

        # Save model
        if (epoch + 1) % config["training"]["save_every"] == 0:
            save_path = os.path.join(config["output"]["save_dir"], f"cemtm_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, save_path)
