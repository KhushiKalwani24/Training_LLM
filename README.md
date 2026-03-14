# 🧠 Training_LLM — Fine-Tuning GPT-2 from Scratch

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Model-GPT--2-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/IDE-PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=white"/>
</p>

<p align="center">
  A clean, modular pipeline for fine-tuning GPT-2 on custom text data using PyTorch and HuggingFace Transformers — built and run entirely in PyCharm.
</p>

---

## 📁 Project Structure

```
Training_LLM/
│
├── data/
│   └── train.txt          # Your custom training text
│
├── config.py              # All hyperparameters in one place
├── dataset.py             # Text loading, tokenization & chunking
├── model.py               # GPT-2 model loader
├── train.py               # Main training loop
├── inference.py           # Generate text from fine-tuned model
├── repeat_data.py         # Utility to expand small datasets
└── requirements.txt       # All dependencies
```

---

## ⚙️ How It Works

```
train.txt  →  dataset.py  →  Tokenize & Chunk  →  DataLoader
                                                        ↓
                                               model.py (GPT-2)
                                                        ↓
                                    config.py → Training Loop (train.py)
                                                        ↓
                                             Fine-tuned Model saved
                                                        ↓
                                           inference.py → Generated Text
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/KhushiKalwani24/Training_LLM.git
cd Training_LLM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your training data
Put any plain text into `data/train.txt` — articles, books, domain knowledge, etc.

### 4. Train the model
```bash
python train.py
```

### 5. Generate text
```bash
python inference.py
```

---

## 🔧 Configuration

All settings are in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `gpt2` | Base model (`gpt2`, `gpt2-medium`) |
| `EPOCHS` | `3` | Training passes over data |
| `BATCH_SIZE` | `4` | Samples per update (lower if OOM) |
| `LEARNING_RATE` | `5e-5` | Optimizer step size |
| `MAX_LENGTH` | `128` | Token context window per chunk |
| `OUTPUT_DIR` | `models/` | Where to save the fine-tuned model |

---

## 📦 Requirements

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
tqdm
```

---

## 🧩 File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Central config — edit hyperparameters here |
| `dataset.py` | Reads `train.txt`, tokenizes, splits into chunks |
| `model.py` | Loads pretrained GPT-2 + moves to GPU/CPU |
| `train.py` | Full training loop: forward → loss → backward → save |
| `inference.py` | Loads saved model, generates text from a prompt |
| `repeat_data.py` | Repeats small datasets to meet minimum token count |

---

## 💡 Key Concepts

| Concept | What it means in this project |
|---------|-------------------------------|
| **Fine-tuning** | Adapting pretrained GPT-2 weights to your custom text |
| **Tokenization** | Converting text → integer token IDs GPT-2 understands |
| **Causal LM** | Model predicts the next token given all previous tokens |
| **AdamW** | Optimizer with weight decay — standard for transformers |
| **LR Warmup** | Gradually increases learning rate to avoid early instability |
| **Gradient Clipping** | Caps gradient norm to prevent exploding gradients |

---

## 🖥️ Hardware

- **GPU recommended** — Tested on NVIDIA GPU (CUDA)
- **CPU fallback** — Works but very slow for large datasets
- Minimum ~4GB VRAM for `gpt2` with `BATCH_SIZE=4`

---

## 📊 Training Output

During training you'll see live loss + learning rate per batch:

```
EPOCH 1/3
Epoch 1: 100%|████████| 28/28 [loss=2.8431, lr=4.80e-05, step=28]
✅ Epoch 1 complete | Avg Loss: 2.6210

EPOCH 2/3
Epoch 2: 100%|████████| 28/28 [loss=2.1073, lr=3.20e-05, step=56]
✅ Epoch 2 complete | Avg Loss: 1.9845
```

Steadily decreasing loss = your model is learning ✅

---

## 🙋 Author

**Khushi Kalwani**  
[![GitHub](https://img.shields.io/badge/GitHub-KhushiKalwani24-181717?style=flat&logo=github)](https://github.com/KhushiKalwani24)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
