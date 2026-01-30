# Quora Question Pairs – Duplicate Detection using Machine Learning

This project focuses on detecting duplicate question pairs from the Quora Question Pairs dataset using classical Machine Learning techniques and extensive feature engineering.

The objective is to determine whether two questions have the same semantic meaning, which is a common real-world NLP problem used in search engines, Q&A platforms, and information retrieval systems.

---

## Problem Statement

Given two questions, identify whether they are duplicates.

This is a binary classification problem:
- 1 → Duplicate questions
- 0 → Non-duplicate questions

---

## Dataset

- Source: Quora Question Pairs Dataset  
- Input Features:
  - question1
  - question2
- Target Variable: is_duplicate

---

## Project Workflow

### 1. Data Loading & Cleaning
- Loaded dataset using Pandas
- Handled missing values
- Removed duplicate and invalid entries

### 2. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Question length statistics
- Duplicate vs non-duplicate patterns

### 3. Text Preprocessing
- Lowercasing
- Removing special characters
- Tokenization
- Stopword removal
- Stemming / Lemmatization (if applicable)

### 4. Feature Engineering

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

### 5. Dimensionality Reduction
- Used t-SNE for 2D visualization of high-dimensional features

### 6. Model Training
- Logistic Regression
- Random Forest
- XGBoost

### 7. Model Evaluation
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

## Project Structure

├── Quora Question Pairs(Using ML).ipynb  
├── README.md

---

## Results

- Successfully distinguishes duplicate and non-duplicate question pairs
- Feature engineering significantly improves model performance
- Fuzzy and distance-based features provide strong semantic signals

Note: Exact metrics may vary depending on configuration.

---

## How to Run

1. Clone the repository:
   git clone https://github.com/your-username/quora-question-pairs-ml.git

2. Install dependencies:
   pip install -r requirements.txt

3. Open the notebook:
   jupyter notebook "Quora Question Pairs(Using ML).ipynb"

4. Run all cells sequentially.

---

## Key Learnings

- NLP feature engineering
- Text similarity and distance metrics
- Classical ML for semantic matching
- High-dimensional data visualization
- Evaluation of imbalanced classification problems

---

## Future Improvements

- Word embeddings (Word2Vec, GloVe)
- Siamese networks
- Transformer-based models (BERT)
- Model deployment

---

## Acknowledgements

- Quora Question Pairs Dataset
- Scikit-learn documentation
