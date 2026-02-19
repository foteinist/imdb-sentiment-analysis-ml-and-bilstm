# IMDB Sentiment Analysis – Classical Machine Learning vs Deep Learning

This project presents a complete sentiment classification pipeline on the IMDB dataset, combining classical machine learning algorithms implemented from scratch with a deep learning architecture based on a Stacked Bidirectional LSTM (BiLSTM).

The goal of this project is to compare traditional ML approaches with neural architectures and analyze their performance using multiple evaluation metrics.

---

## Project Overview

The project is divided into three main parts:

### Part A – Custom Machine Learning Implementations
- Bernoulli Naive Bayes (from scratch)
- ID3 Decision Tree (from scratch)
- Random Forest (ensemble of custom ID3 trees)
- Feature selection using Information Gain
- Learning curves (Precision, Recall, F1)

### Part B – Comparison with scikit-learn
- Bernoulli Naive Bayes (scikit-learn)
- Random Forest (scikit-learn)
- Detailed metric comparison between custom and library implementations

### Part C – Deep Learning Model
- Stacked Bidirectional LSTM (2 layers)
- Global Max Pooling
- Pretrained GloVe embeddings (100 dimensions)
- Adam optimizer
- Model selection based on validation F1-score
- Evaluation on test dataset

---

## Dataset

- IMDB Movie Reviews Dataset
- 25,000 training samples
- 25,000 test samples
- Binary sentiment classification (positive / negative)

For the deep learning model:
- Training set split: 80% training / 20% validation
- Sequences padded to length 500
- Vocabulary size: 10,000 words

---

## Project Structure
```

imdb-sentiment-analysis-ml-and-bilstm/
│
├── part_a_b/
│   ├── main.py
│   └── main2.py
│
├── part_c/
│   └── my_rnn_model.py
│
├── report.pdf
├── requirements.txt
├── README.md
└── RUNNING.md


---

## Classical ML Implementation Details

### Bernoulli Naive Bayes
- Binary feature representation (0/1)
- Laplace smoothing
- Log-likelihood computation
- Custom `predict` and `predict_proba`

### ID3 Decision Tree
- Entropy-based Information Gain
- Recursive tree construction
- Depth and sample-based stopping criteria

### Random Forest
- Bootstrap sampling
- Ensemble voting
- Configurable number of trees and depth

### Evaluation Metrics
- Precision
- Recall
- F1-score
- Macro & Micro averages
- Learning curves

---

## Deep Learning Architecture (Stacked BiLSTM)

- Embedding layer initialized with pretrained GloVe vectors (100d)
- 2-layer Bidirectional LSTM
- Hidden dimension: 128
- Global Max Pooling
- Dropout (0.5)
- Fully connected output layer
- Sigmoid activation (binary classification)

### Training Configuration
- Loss: Binary Cross Entropy
- Optimizer: Adam (LR = 0.001)
- Batch size: 32
- Epochs: 20
- Best model selected using validation F1-score

---

## Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Dependencies

```
numpy
scikit-learn
matplotlib
tensorflow
torch
```

---

## Running the Project

### Part A & B – Classical ML

```bash
python part_a_b/main.py
```

The script allows:
- Vocabulary size configuration
- Removal of most frequent and rare words
- Selection of top Information Gain features

Learning curves and evaluation metrics will be displayed.

---

### Part C – Stacked BiLSTM

Before running:

Download GloVe embeddings (100d):
https://nlp.stanford.edu/projects/glove/

Place the file:

```
glove.6B.100d.txt
```

inside the project root directory.

Run:

```bash
python part_c/my_rnn_model.py
```

The script:
- Trains the model
- Saves best model (`best_model.pth`)
- Plots training & validation loss
- Prints test Precision, Recall, F1-score

---

## Results Summary

Key findings:

- Custom Bernoulli Naive Bayes achieves performance very close to scikit-learn.
- Custom Random Forest underperforms compared to optimized scikit-learn implementation.
- Stacked BiLSTM significantly outperforms classical ML methods.
- Validation-based model selection improves generalization.
- Deep learning model demonstrates balanced macro & micro performance.

---

## Technical Skills Demonstrated

- Machine Learning algorithms implemented from scratch
- Entropy & Information Gain computation
- Ensemble methods
- Text preprocessing & feature engineering
- Model evaluation & metric analysis
- Learning curve visualization
- Deep Learning with PyTorch
- Pretrained embeddings integration
- Model selection & overfitting analysis

---


## Authors

- Konstantina Karapetsa

- Foteini Sotiropoulou

Developed as part of coursework in Artificial Intelligence.


