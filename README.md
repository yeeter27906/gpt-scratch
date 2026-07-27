# 🧠 minGPT

A minimal, clean, and highly interpretable PyTorch implementation of OpenAI's [GPT](https://github.com/openai/gpt-2). 

While many large language model implementations are sprawling and difficult to parse, minGPT distills the architecture down to its essentials. The core Transformer model is implemented in approximately 300 lines of code (`mingpt/model.py`). The architecture relies on feeding a sequence of indices into a [Transformer](https://arxiv.org/abs/1706.03762) to output a probability distribution over the next index. The primary complexity is isolated to efficient batching across examples and sequence lengths.

> **⚠️ Project Status (Jan 2023):** minGPT is currently in a semi-archived state. Because it is widely referenced in educational materials, breaking changes are avoided. For a more modern, hackable framework focused on runtime efficiency and reproducing medium-sized industry benchmarks, check out the successor project: **[nanoGPT](https://github.com/karpathy/nanoGPT)**.

## 📁 Repository Architecture

The core library is isolated to three cleanly separated files:
*   `mingpt/model.py`: The Transformer model definition.
*   `mingpt/bpe.py`: A refactored Byte Pair Encoder for translating text to integer sequences (matching OpenAI's GPT).
*   `mingpt/trainer.py`: GPT-independent PyTorch boilerplate for model training.

**Included Projects & Demos:**
*   `projects/adder`: Trains a GPT from scratch to perform mathematical addition (inspired by GPT-3).
*   `projects/chargpt`: Trains a character-level language model on a provided text file.
*   `demo.ipynb`: A minimal notebook demonstrating a simple sorting task using the `GPT` and `Trainer` classes.
*   `generate.ipynb`: A guide on loading a pre-trained GPT-2 model for prompt-based text generation.

## 🚀 Quick Start

### Installation

To integrate the library into a project, clone and install locally:

```bash
git clone [https://github.com/karpathy/minGPT.git](https://github.com/karpathy/minGPT.git)
cd minGPT
pip install -e .
