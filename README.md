# Small GPT – Mini Transformer

A small GPT-style decoder-only Transformer implemented from scratch using PyTorch. The model is trained on the Tiny Shakespeare dataset to learn character-level next-token prediction and generate Shakespeare-like text.

## 📌 Project Overview

This project demonstrates the fundamental architecture and working of a GPT-style language model at a small scale.

The model learns to predict the next character based on the previous sequence of characters. After training, it can generate new text autoregressively from a given prompt.

## 🚀 Features

* Character-level tokenization
* Train/validation dataset split
* Token embeddings
* Positional embeddings
* Causal self-attention
* Multi-head self-attention
* Transformer blocks
* Layer normalization
* Residual connections
* Feed-forward MLP with GELU activation
* Cross-entropy loss
* AdamW optimizer
* Training and validation loss visualization
* Autoregressive text generation
* Next-character probability analysis

## 🧠 Model Architecture

The model follows a decoder-only Transformer architecture:

```text
Input Text
    ↓
Character Tokenization
    ↓
Token Embedding + Positional Embedding
    ↓
Transformer Block × 4
    ↓
Causal Multi-Head Self-Attention
    ↓
Feed-Forward MLP
    ↓
Layer Normalization + Residual Connections
    ↓
Final Layer Normalization
    ↓
Linear Language Model Head
    ↓
Logits
    ↓
Next-Character Probabilities
```

## 📚 Dataset

The model is trained on the **Tiny Shakespeare** text corpus.

The dataset contains Shakespeare's works and is used here as a small text corpus for demonstrating language-model training and text generation.

## ⚙️ Technical Details

| Component           | Configuration      |
| ------------------- | ------------------ |
| Framework           | PyTorch            |
| Tokenization        | Character-level    |
| Embedding Dimension | 128                |
| Transformer Layers  | 4                  |
| Attention Heads     | 4                  |
| Context Length      | 128                |
| Batch Size          | 64                 |
| Dropout             | 0.1                |
| Learning Rate       | 3e-4               |
| Optimizer           | AdamW              |
| Loss Function       | Cross-Entropy Loss |
| Training Iterations | 3000               |

## 🔄 Training Process

The model is trained using next-character prediction.

For an input sequence:

```text
ROMEO
```

the corresponding targets are shifted by one position:

```text
Input:  R O M E O
Target: O M E O ...
```

The model predicts the next character at every position. Cross-entropy loss measures the difference between the predicted probabilities and the actual target characters.

The loss is then used for backpropagation, and AdamW updates the model parameters.

## ✨ Text Generation

After training, the model can generate text from a starting prompt such as:

```text
ROMEO:
```

The model predicts a probability distribution for the next character, samples a character, adds it to the context, and repeats the process.

This makes the generation process autoregressive.

## 🛠️ Technologies Used

* Python
* PyTorch
* Jupyter Notebook
* Matplotlib
* Tiny Shakespeare Dataset

## 📁 Repository Structure

```text
small-gpt/
│
├── GenAI.ipynb
├── README.md
└── .gitignore
```

## 🎯 Learning Outcomes

This project provides practical understanding of:

* Transformer architecture
* Self-attention
* Causal masking
* Multi-head attention
* Embeddings
* Positional information
* Residual connections
* Layer normalization
* Language-model training
* Next-token prediction
* Autoregressive text generation

