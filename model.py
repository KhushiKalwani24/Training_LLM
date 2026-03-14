# model.py
# ─────────────────────────────────────────────────────────────
# Loads a pretrained causal language model from HuggingFace.
# Fine-tuning starts from pretrained weights (transfer learning),
# which is MUCH faster than training from scratch.
# ─────────────────────────────────────────────────────────────

import torch
from transformers import AutoModelForCausalLM


def load_model(config):
    """
    Loads a pretrained causal LM and moves it to the target device.

    AutoModelForCausalLM automatically picks the right architecture
    based on the model name (GPT-2, GPT-J, LLaMA, etc.)

    Args:
        config: Config object with MODEL_NAME and DEVICE

    Returns:
        model: The loaded model on the correct device
    """

    print(f"[Model] Loading '{config.MODEL_NAME}'...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        # torch_dtype=torch.float16,  # Uncomment for 16-bit (saves VRAM)
    )

    # ── Move model to GPU/CPU ──────────────────────────────
    device = torch.device(config.DEVICE
                          if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)

    # ── Count trainable parameters ─────────────────────────
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {num_params:,}")
    print(f"[Model] Running on: {device}")

    return model, device