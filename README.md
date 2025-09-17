# Binary Classification with Neural Networks (PyTorch)

A PyTorch-based binary classification model trained on a modified Census Income dataset to predict whether an individual earns more than $50K annually.

## 🧾 Dataset

- File: `income.csv`
- Rows: 30,000
- Features:
  - Categorical: `sex`, `education`, `marital-status`, `workclass`, `occupation`
  - Continuous: `age`, `hours-per-week`
  - Target: `label` (0: <=50K, 1: >50K)

## 🧠 Model Architecture

- Embedding layers for categorical inputs
- Batch normalization for continuous inputs
- 1 hidden layer (50 units) with ReLU
- Dropout (p=0.4)
- Final layer: 2 output units (binary classification)

## 🛠️ Setup

```bash
pip install torch pandas matplotlib scikit-learn
