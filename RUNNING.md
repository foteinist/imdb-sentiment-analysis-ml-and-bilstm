# Running Guide

This document provides step-by-step instructions for running the IMDB Sentiment Analysis project.

---

# 1. Environment Setup

It is recommended to use a virtual environment.

## Create virtual environment

```bash
python -m venv venv
```

## Activate environment

Mac/Linux:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

---

# 2. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install manually:

```bash
pip install numpy scikit-learn matplotlib torch tensorflow
```

---

# 3. Running Classical ML (Part A & B)

Navigate to the project root directory and run:

```bash
python part_a_b/main.py
```

Optional (if you want the comparison script):

```bash
python part_a_b/main2.py
```

### What happens:

- Text preprocessing
- Feature extraction (binary representation)
- Training:
  - Bernoulli Naive Bayes (custom)
  - ID3 Decision Tree
  - Random Forest
- Evaluation using:
  - Precision
  - Recall
  - F1-score
- Learning curve generation

Results are printed in the terminal.

---

# 4. Running Deep Learning Model (Part C – BiLSTM)

## Step 1: Download GloVe Embeddings

Download from:

https://nlp.stanford.edu/projects/glove/

Choose:
```
glove.6B.zip
```

Extract and place:

```
glove.6B.100d.txt
```

inside the project root directory.

---

## Step 2: Run the Model

```bash
python part_c/my_rnn_model.py
```

### What happens:

- IMDB dataset loading
- Tokenization & padding (max length = 500)
- Vocabulary size = 10,000
- Pretrained GloVe embedding loading
- 2-layer Bidirectional LSTM training
- Validation-based model selection
- Best model saved as:

```
best_model.pth
```

- Training & validation loss plots generated
- Test Precision, Recall, F1-score printed

---

# 5. Expected Runtime

- Classical ML: Few minutes (CPU)
- BiLSTM training: 10–30 minutes (CPU)
- Faster with GPU (if available)

---

# 6. Common Issues

### 1. torch not found
Install:
```bash
pip install torch
```

### 2. GloVe file not found
Ensure:
```
glove.6B.100d.txt
```
is placed in the project root directory.

### 3. CUDA not available
The model will automatically run on CPU if GPU is not detected.

---

# 7. Output Files

After running the deep learning model:

- `best_model.pth` → Saved trained model
- Loss plots → Training & validation curves

---

# 8. Reproducibility

For reproducible results, ensure:

- Same Python version (recommended 3.9+)
- Same package versions
- Fixed random seeds (if enabled in code)

---

If you encounter issues, verify paths and dependencies before running again.
