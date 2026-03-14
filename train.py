# train.py
# ─────────────────────────────────────────────────────────────
# Orchestrates the full training pipeline:
#   1. Load config, tokenizer, dataset, model
#   2. Set up optimizer & learning rate scheduler
#   3. Run training loop (forward → loss → backward → update)
#   4. Save checkpoints & final model
# ─────────────────────────────────────────────────────────────

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm  # Progress bars

from config import Config
from dataset import build_tokenizer_and_dataset
from model import load_model


def train():
    # ══════════════════════════════════════════════════════
    # 1. SETUP
    # ══════════════════════════════════════════════════════
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)   # Create output folder

    # ── Load tokenizer + dataset ───────────────────────────
    tokenizer, dataset = build_tokenizer_and_dataset(config)

    # ── DataLoader: batches + shuffling ────────────────────
    # shuffle=True randomizes sample order each epoch
    # num_workers=2 loads batches in background (faster on GPU machines)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,    # Set to 2-4 if not on Windows
        pin_memory=True,  # Faster CPU→GPU transfer
    )
    print(f"[DataLoader] {len(dataloader)} batches per epoch")

    # ── Load model ─────────────────────────────────────────
    model, device = load_model(config)
    model.train()   # Put model in training mode (enables dropout, etc.)

    # ══════════════════════════════════════════════════════
    # 2. OPTIMIZER & SCHEDULER
    # ══════════════════════════════════════════════════════

    # AdamW = Adam + decoupled weight decay (standard for transformers)
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Total number of gradient update steps across all epochs
    total_steps = len(dataloader) * config.EPOCHS

    # Linear warmup then linear decay of LR
    # Warmup prevents large, destabilizing updates at the start
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    print(f"\n[Train] Starting training for {config.EPOCHS} epochs "
          f"({total_steps} total steps)\n")

    # ══════════════════════════════════════════════════════
    # 3. TRAINING LOOP
    # ══════════════════════════════════════════════════════
    global_step = 0          # Counts total optimizer steps
    running_loss = 0.0       # Accumulates loss for logging

    for epoch in range(1, config.EPOCHS + 1):

        print(f"{'='*50}")
        print(f"  EPOCH {epoch} / {config.EPOCHS}")
        print(f"{'='*50}")

        # tqdm wraps the dataloader to show a live progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

        for batch in progress_bar:

            # ── Move batch to GPU/CPU ──────────────────────
            input_ids = batch["input_ids"].to(device)   # [B, L]
            labels    = batch["labels"].to(device)       # [B, L]

            # ── Forward pass ───────────────────────────────
            # The model predicts the next token at each position.
            # When labels are provided, it also computes cross-entropy loss
            # automatically (comparing predicted vs actual next token).
            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )
            loss = outputs.loss   # Scalar: average cross-entropy over batch

            # ── Backward pass ──────────────────────────────
            # Zero out old gradients — PyTorch accumulates them by default
            optimizer.zero_grad()

            # Compute gradients via backpropagation
            loss.backward()

            # Gradient clipping: cap gradient norm to prevent exploding gradients
            # (a common problem in transformer training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ── Optimizer step ─────────────────────────────
            optimizer.step()     # Update model weights
            scheduler.step()     # Update learning rate

            # ── Logging ────────────────────────────────────
            global_step += 1
            running_loss += loss.item()

            if global_step % config.LOG_INTERVAL == 0:
                avg_loss = running_loss / config.LOG_INTERVAL
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr":   f"{current_lr:.2e}",
                    "step": global_step,
                })
                running_loss = 0.0   # Reset accumulator

            # ── Save checkpoint ────────────────────────────
            if global_step % config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    config.OUTPUT_DIR, f"checkpoint-step-{global_step}"
                )
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"\n[Checkpoint] Saved to {checkpoint_path}")

        print(f"\n[Epoch {epoch}] Complete. Loss: {loss.item():.4f}\n")

    # ══════════════════════════════════════════════════════
    # 4. SAVE FINAL MODEL
    # ══════════════════════════════════════════════════════
    final_path = os.path.join(config.OUTPUT_DIR, "final-model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Training complete! Final model saved to: {final_path}")


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    train()