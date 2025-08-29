# EMNLP-GAttention

This repository presents the implementations and configurations proposed to evaluate different variants of the **GAttention** and **Multi-head GAttention** mechanisms for abusive language detection.

## Project Structure

- **Utilities/**  
  Contains all the necessary classes to run the project, including data loading, tokenization, model architectures, and the complete experimental pipeline.

- **Datasets/**  
  Contains the evaluation datasets used in our experiments.  
  *(Note: We are not the owners of these datasets. For details and references, please refer to our paper.)*

## Citation

If you use this repository or any part of the provided code, please cite the following paper:

```bibtex
@inproceedings{
vasquez2025gattention,
title={{GA}ttention: Gated Attention for the Detection of Abusive Language},
author={Horacio Jarqu{\'\i}n V{\'a}squez and Hugo Jair Escalante and Manuel Montes and Mario Ezra Aragon},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025}
}
```

## 🚀 Execution

To compile and run the project, you must execute the `main.py` file with the following parameters:

### Compilation Parameters

- `--dataset` → Indicates the training dataset.  
  Possible values:
  - `SEM`
  - `AMI`
  - `HAS`

- `--subtask` → Defines the training subtask.  
  Possible values:
  - `A`
  - `B`

- `--model` → Specifies the model used for training.  
  Possible values:
  - `"roberta-base"`
  - `"nghuyong/ernie-2.0-base-en"`
  - `"bert-base-uncased"`
  - `"distilbert-base-uncased"`

- `--architecture_mode` → Defines the architecture used in training.  
  Possible values:
  - `MHG` → Multi-head GAttention mechanism (12 heads, best configuration).  
  - `GAT` → Standard GAttention mechanism.  
  - `NON` → No attention mechanism included; only fine-tuning with the selected model.

---

## 📌 Example Usage

If you want to train on **subtask B** of the **AMI dataset**, using the `nghuyong/ernie-2.0-base-en` model with the **GAttention** architecture, the command would be:

```bash
python3 main.py --dataset AMI --subtask B --model nghuyong/ernie-2.0-base-en --architecture_mode GAT