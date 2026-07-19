<div align="center">

# 🧠 LLM Engineering & Advanced NLP

### Building Modern Large Language Model Systems from Fundamentals to Production

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)]()
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-success)]()

---

**End-to-End LLM Engineering • Transformer Architecture • Fine-Tuning • Quantization • RLHF • Production Deployment**

</div>

---

# 📖 Overview

This repository provides a comprehensive engineering journey covering the complete lifecycle of **Large Language Models (LLMs)**.

Unlike repositories focusing only on model training, this project explores how modern LLM systems are **designed, optimized, aligned and deployed** in production environments.

The repository progresses from **classical Natural Language Processing** concepts to **Transformer architectures**, **parameter-efficient fine-tuning**, **model optimization**, **alignment techniques**, and finally **microservice-based deployment**.

---

# 🏛 Repository Architecture

```text
                    Large Language Models

                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
      NLP Basics      Transformer        Deployment
         │                 │                  │
 Text Processing     Self Attention     FastAPI APIs
 TF-IDF              Encoder            Docker
 Word2Vec            Decoder            Gateway
         │                 │                  │
         └──────────────┬─────────────────────┘
                        │
                  Fine-Tuning
                        │
                Quantization
                        │
                 RLHF & DPO
```

---

# 📂 Project Structure

| Module | Description | Engineering Focus |
|:--|:--|:--|
| **01 NLP Basics** | Text preprocessing | Tokenization, Cleaning |
| **02 BoW & N-Gram** | Classical NLP | Feature Engineering |
| **03 Word Embeddings** | Distributed Representations | Word2Vec |
| **04 Transformer** | Transformer Architecture | Encoder–Decoder |
| **05 Self Attention** | Attention Mechanism | Query, Key, Value |
| **06 Autoregressive Generation** | Text Generation | Greedy Search, Temperature |
| **07 Fine-Tuning** | Transfer Learning | PEFT, LoRA |
| **08 Quantization** | Model Optimization | INT8 / 4-bit |
| **09 RLHF & DPO** | Alignment | Preference Optimization |
| **10 Microservices** | Production Serving | FastAPI + Docker |

---

# ⚙ Engineering Pipeline

```text
Raw Text
    │
    ▼
Text Preprocessing
    │
    ▼
Embeddings
    │
    ▼
Transformer
    │
    ▼
Self Attention
    │
    ▼
Autoregressive Generation
    │
    ▼
Fine-Tuning
    │
    ▼
Quantization
    │
    ▼
RLHF / DPO
    │
    ▼
FastAPI Deployment
```

---

# 🛠 Tech Stack

## Programming

- Python

## Deep Learning

- PyTorch
- Hugging Face Transformers
- PEFT
- TRL

## NLP

- NLTK
- spaCy
- Scikit-Learn

## Model Optimization

- LoRA
- QLoRA
- BitsAndBytes
- Quantization
- Mixed Precision

## Backend

- FastAPI
- REST API

## DevOps

- Docker

---

# 🧩 Core Engineering Topics

## Natural Language Processing

- Text Cleaning
- Tokenization
- Lemmatization
- Stopword Removal
- TF-IDF
- N-Gram
- Word2Vec

---

## Transformer Architecture

- Positional Encoding
- Multi-Head Attention
- Feed Forward Networks
- Residual Connections
- Layer Normalization
- Encoder
- Decoder

---

## Large Language Models

- Autoregressive Generation
- Temperature Sampling
- Top-K Sampling
- Top-P Sampling
- Beam Search
- Logits Processing

---

## Fine-Tuning

- Transfer Learning
- PEFT
- LoRA
- QLoRA
- Hugging Face Trainer

---

## Optimization

- Quantization
- Mixed Precision
- Memory Optimization
- VRAM Reduction
- Low-bit Inference

---

## Alignment

- RLHF
- DPO
- Human Preference Learning
- Reward Modeling

---

## Production

- FastAPI
- REST API
- Docker
- API Gateway
- Model Serving
- Scalable Deployment

---

# 🎯 Learning Outcomes

After completing this repository you will understand:

- Classical NLP techniques
- Transformer architecture from scratch
- Self-Attention mathematics
- Large Language Model inference
- Fine-tuning modern LLMs
- PEFT methods
- LoRA & QLoRA
- Quantization techniques
- RLHF and DPO
- Production-ready LLM deployment
- FastAPI serving architecture
- Docker-based deployment

---

# 🚀 Getting Started

Clone the repository

```bash
git clone https://github.com/username/LLM-Engineering.git

cd LLM-Engineering
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run notebooks individually according to the learning path.

---

# 📈 Engineering Philosophy

This repository follows a **System-Centric AI Engineering** approach rather than a **Model-Centric** perspective.

The objective is not only to train language models but also to understand how modern AI systems are:

- Designed
- Optimized
- Fine-tuned
- Evaluated
- Aligned
- Served
- Scaled

for real-world production environments.

---

# 📚 References

- Attention Is All You Need
- Hugging Face Transformers
- PEFT
- TRL
- PyTorch Documentation
- FastAPI Documentation

---

<div align="center">

### ⭐ If you find this repository useful, consider giving it a star.

**Made with ❤️ by Şevval Mıkçı**

</div>
