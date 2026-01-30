# Quora Question Pairs – Duplicate Detection using Machine Learning

This repository contains multiple notebooks focused on solving the **Quora Question Pairs duplicate detection problem** using **classical Machine Learning**, extensive **feature engineering**, and an **optimized XGBoost model**.

The objective is to determine whether two questions are semantically equivalent.

---

## Problem Statement

Given two questions, predict whether they are **duplicates**.

This is a binary classification problem:
- `1` → Duplicate questions  
- `0` → Non-duplicate questions  

---

## Dataset

- **Source:** Quora Question Pairs Dataset  
- **Input Columns:**
  - `question1`
  - `question2`
- **Target Column:** `is_duplicate`

---

## Notebooks Included

### 1. Quora Question Pairs (Using ML)
**File:** `Quora Question Pairs(Using ML).ipynb`

Covers:
- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Text preprocessing
- Feature engineering
- Baseline ML models

---

### 2. XGBoost Optimised Model
**File:** `XGBoost_Optimised.ipynb`

Covers:
- Training an optimized XGBoost classifier
- Hyperparameter tuning
- Performance improvement over baseline models
- Detailed model evaluation

---

## Project Workflow

### Data Cleaning
- Handling missing values
- Removing duplicates
- Validating text fields

### Text Preprocessing
- Lowercasing
- Removing special characters
- Tokenization
- Stopword removal

### Feature Engineering

#### Basic Features
- Question length
- Word count differences
- Common word counts

#### Distance-Based Features
- Levenshtein distance
- Ratio-based similarity scores

#### Fuzzy Matching Features
- Fuzzy ratio
- Partial ratio
- Token sort ratio
- Token set ratio

#### Vector-Based Features
- TF-IDF representations

---

### Dimensionality Reduction
- t-SNE visualization for high-dimensional feature space

---

### Model Training
- Logistic Regression
- Random Forest
- XGBoost (Optimised)

---

### Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- RapidFuzz / Distance
- Matplotlib
- Seaborn

---

## Repository Structure

```
├── Quora Question Pairs(Using ML).ipynb
├── XGBoost_Optimised.ipynb
├── README.md
```

---

## Results

- Feature engineering significantly boosts performance
- Optimized XGBoost model outperforms baseline ML approaches
- Fuzzy and distance-based features capture semantic similarity effectively

*(Exact metrics may vary based on configuration.)*

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/quora-question-pairs-ml.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

4. Run notebooks in sequence:
   - `Quora Question Pairs(Using ML).ipynb`
   - `XGBoost_Optimised.ipynb`

---

## Key Learnings

- Practical NLP feature engineering
- Text similarity and distance metrics
- Boosting-based ML models for NLP
- Visualization of high-dimensional data
- Evaluation of imbalanced datasets

---

## Future Improvements

- Word embeddings (Word2Vec, GloVe)
- Siamese neural networks
- Transformer-based approaches (BERT)
- Model deployment

---

## Acknowledgements

- Quora Question Pairs Dataset
- Scikit-learn documentation
- XGBoost documentation
