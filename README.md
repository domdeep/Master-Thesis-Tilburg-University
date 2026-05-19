# Self-Supervised Learning for Plant Disease Classification: A Comparative Study of SimCLR and BYOL

MSc Thesis, Data Science and Society, Tilburg University, 2025

###Grade: 8/10

This repository contains the code and resources for my Master's thesis submitted in partial fulfillment of the requirements for the MSc in Data Science and Society at Tilburg University (June 2025).

The study benchmarks two leading self-supervised learning methods, SimCLR (contrastive) and BYOL (non-contrastive), against a fully supervised baseline for plant disease classification, using ConvNeXt-Tiny as the backbone architecture and the CCMT dataset.

---

## Overview

Manual annotation of agricultural image data is costly and time-consuming. This thesis investigates whether self-supervised learning (SSL) can serve as a practical alternative by leveraging unlabeled data for plant disease detection. The study is the first direct comparison of SimCLR and BYOL within a consistent framework for this domain.

---

## Research Questions

**Main RQ:** How effectively can modern CNN architectures (ConvNeXt-Tiny, EfficientNet-B0, DenseNet-121) serve as supervised baselines for plant disease classification?

**SRQ1:** How does SimCLR improve classification using the best-performing CNN?

**SRQ2:** How does BYOL improve classification using the best-performing CNN?

**SRQ3:** How do SimCLR and BYOL compare in label efficiency against the supervised baseline?

---

## Dataset

The CCMT Dataset for Crop Pest and Disease Detection (Mensah et al., 2023) was used. It contains 24,881 raw images across 22 classes covering Cassava, Cashew, Maize, and Tomato diseases, collected from farms in Ghana and licensed under CC BY 4.0.

After preprocessing (removal of corrupted files, duplicates, and unrelated insect classes), 17 disease classes were retained for classification.

---

## Models

**Supervised Baselines**

- EfficientNet-B0
- ConvNeXt-Tiny (best performer, selected as SSL backbone)
- DenseNet-121

**Self-Supervised Pretraining**

- SimCLR: contrastive learning with NT-Xent loss and a strong augmentation pipeline
- BYOL: non-contrastive learning with online and target network asymmetry and EMA updates

**Ablation**

- SimCLR + ImageNet-1K initialization
- BYOL + ImageNet-1K initialization

---

## Results

| Model | F1-score | Precision | Recall | Accuracy |
|---|---|---|---|---|
| ConvNeXt-Tiny (Supervised) | 0.9286 | 0.9331 | 0.9253 | 0.9415 |
| ConvNeXt-Tiny (SimCLR) | 0.8675 | 0.8576 | 0.8818 | 0.8786 |
| ConvNeXt-Tiny (BYOL) | 0.7721 | 0.7657 | 0.7230 | 0.8051 |
| ConvNeXt-Tiny (SimCLR + ImageNet-1K) | 0.9311 | 0.9285 | 0.9354 | 0.9376 |
| ConvNeXt-Tiny (BYOL + ImageNet-1K) | 0.8042 | 0.7907 | 0.8351 | 0.8217 |

SimCLR pretrained with ImageNet-1K weights slightly outperformed the fully supervised baseline, demonstrating the value of combining transfer learning with self-supervised pretraining. BYOL showed reasonable results but underperformed SimCLR throughout, particularly in low-label scenarios.

---

## Pipeline

- Framework: PyTorch and PyTorch Lightning
- Hyperparameter tuning: Optuna (TPE algorithm, 15 trials per model)
- Validation: 5-fold stratified cross-validation, no data leakage
- SSL implementation: Adapted from the Lightly library
- Evaluation: Macro F1-score, precision, recall, accuracy, confusion matrices, t-SNE visualizations
- Label efficiency: Tested at 5%, 10%, 20%, 40%, 60%, 80%, and 100% of labeled data

---

## Tech Stack

| Package | Version |
|---|---|
| PyTorch | 2.5.1+cu121 |
| PyTorch Lightning | 2.5.1 |
| Lightly | 1.5.20 |
| Optuna | 4.2.1 |
| Scikit-learn | 1.6.1 |
| NumPy | 2.2.5 |
| Pandas | 2.2.3 |
| Matplotlib | 3.9.2 |

---

## Acknowledgements

This thesis was supervised by Dr. Gorkem Saygili and Dr. Kyana van Eijndhoven at Tilburg University, School of Humanities and Digital Sciences. SSL implementations were adapted from Lightly's official SimCLR and BYOL implementations.
