# inference.py
# ─────────────────────────────────────────────────────────────
# Load your fine-tuned model and generate text from a prompt.
# Run this after training to verify the model learned something.
# ─────────────────────────────────────────────────────────────

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(prompt: str, model_path: str = "output/final-model",
             max_new_tokens: int = 200):
    """
    Loads a saved model and generates text continuation for a prompt.

    Args:
        prompt        : Starting text for generation
        model_path    : Path to your saved fine-tuned model
        max_new_tokens: How many new tokens to generate
    """

    # ── Load model + tokenizer from saved path ─────────────
    print(f"[Inference] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # Disable dropout for inference

    # ── Tokenize the input prompt ──────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ── Generate tokens ────────────────────────────────────
    with torch.no_grad():  # Don't compute gradients (saves memory)
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Sampling = creative, varied output
            temperature=0.8,  # Lower = conservative, higher = creative
            top_p=0.95,  # Nucleus sampling: keeps top 95% probability mass
            repetition_penalty=1.2,  # Penalize repeating the same tokens
            pad_token_id=tokenizer.eos_token_id,
        )

    # ── Decode and print ───────────────────────────────────
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n" + "─" * 50)
    print("Generated Output:")
    print("─" * 50)
    print(generated_text)
    return generated_text


if __name__ == "__main__":
    generate(prompt="Once upon a time in a land far away,")