# config.py
# ─────────────────────────────────────────────────────────────
# All training hyperparameters in one place.
# Adjust these without touching training logic.
# ─────────────────────────────────────────────────────────────

class Config:
    # ── Model ──────────────────────────────────────────────
    MODEL_NAME = "gpt2"          # Pretrained base model from HuggingFace
                                  # Options: "gpt2", "gpt2-medium", "distilgpt2"
                                  # For custom: swap with your architecture

    MAX_LENGTH = 128              # Max token length per training sample
                                  # Longer = more context, more VRAM

    # ── Training ───────────────────────────────────────────
    EPOCHS = 3                    # How many full passes over the dataset
    BATCH_SIZE = 4                # Samples per gradient update
                                  # Lower if you get CUDA out-of-memory errors
    LEARNING_RATE = 5e-5          # Step size for optimizer
                                  # Too high → unstable, too low → slow
    WARMUP_STEPS = 100            # Gradually ramp up LR at the start
                                  # Helps avoid early instability
    WEIGHT_DECAY = 0.01           # L2 regularization to prevent overfitting

    # ── Paths ──────────────────────────────────────────────
    DATA_PATH = "data/train.txt"  # Path to your raw training text
    OUTPUT_DIR = "output/"        # Where to save checkpoints & final model

    # ── Hardware ───────────────────────────────────────────
    DEVICE = "cuda"               # "cuda" for GPU, "cpu" for CPU-only
                                  # Training on CPU is very slow for LLMs

    # ── Logging ────────────────────────────────────────────
    LOG_INTERVAL = 50             # Print loss every N steps
    SAVE_INTERVAL = 500           # Save checkpoint every N steps