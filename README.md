# ADM Project — Anomaly Detection in Network Flows

This repository contains the implementation for the project **Anomaly Detection in Network Flows**. The project explores multiple machine learning and deep learning approaches for detecting anomalies in network traffic, along with explainability (XAI) analysis of the trained models.

## Overview

The project consists of four approaches, along with a preprocessing file for dataset.

## Preprocessing

Dataset preprocessing is implemented in [`preprocessing.ipynb`](./preprocessing.ipynb).

## Approach 1: Machine Learning Models

Implemented in [`Approach_1_Final.ipynb`](./Approach_1_Final.ipynb).

This approach trains and evaluates four machine learning models:

- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier

**Library used:** 
- `pandas`
- `numpy`
-`sklearn`

## Approach 2: Artificial Neural Networks

Implemented in [`Approach_2_Final.ipynb`](./Approach_2_Final.ipynb).

This approach explores three different Artificial Neural Network architectures/hyperparameter configurations.

**Library used:** 
- `pandas`
- `numpy`
-`tensorflow`

## Approach 3: Explainable AI (XAI) Analysis

Implemented in [`Approach_3_Final.ipynb`](./Approach_3_Final.ipynb).

This approach takes the four machine learning models from Approach 1 and applies two XAI techniques to understand the nature of predictions:

- **Partial Dependence Plot (PDP)**
- **Individual Conditional Expectation (ICE)**

These techniques are evaluated using four XAI metrics (To calculate PDP-based Faithfulness (correlation with permutation importance) we calculated Permuation Importance as well):

1. **PDP-based Faithfulness** — correlation with Permutation Importance (Permutation Importance is also computed as part of this evaluation)
2. **Sparsity** — threshold = `0.01 × max(pdp_importance_series)`
3. **Randomization Test Correlation**
4. **Sensitivity**

**Key libraries used:**
- `sklearn`
- `scipy`
- `matplotlib`

## Approach 4: Two-Tier Behavioral Cascade

Implemented in 3 files [`Approach_4_Final_Training.py`](./Approach_4_Final_Training.py), [`Approach_4_Final_Evaluation.py`](./Approach_4_Final_Evaluation.py), [`Approach_4_Final_Inference.py`](./Approach_4_Final_Inference.py).

This approach implements a **Two-Tier Behavioral Cascade** architecture designed to detect zero-day attacks and mitigate cross-network domain shift:

- **Tier 1 — Geometric Filter (LSH-DBSCAN):** Drops massive volumetric anomalies using a composite deviation score.
- **Tier 2 — Gradient Boosting Arbiter:** Evaluates a pruned 32-dimensional behavioral vector to catch stealthy, shape-shifting exploits.

### Evaluation

The pipeline is evaluated on both:
- **In-distribution data:** CIC-IDS2017
- **Unseen cross-domain zero-day traffic:** CIC-IDS2018

Using the following metrics:
- Overall Accuracy
- Malign Accuracy (TPR)
- Benign Accuracy (TNR)
- False Positive Rate (FPR)
- Per-flow latency

### Real-Time Inference & Local XAI

A dedicated real-time inference script is included to ensure operational transparency. It outputs live detection results alongside local XAI telemetry, explicitly logging the top feature deviations for every flagged flow.

**Key libraries used:**
- `pandas`
- `numpy`
- `scipy`
- `sklearn`
- `joblib`

## Repository Structure

```
.
├── preprocessing.ipynb        # Dataset preprocessing
├── Approach_1_Final.ipynb     # Classical ML models (Logistic Regression, GNB, Decision Tree, Random Forest)
├── Approach_2_Final.ipynb     # Artificial Neural Network architectures
├── Approach_3_Final.ipynb     # XAI analysis (PDP, ICE) on Approach 1 models
├── Approach_4_Final_Training.py  # Training of the dataset for Approach-4
├── Approach_4_Final_Evaluation.py # Using evaluation metrics
├── Approach_4_Final_Inference.py  # Getting inferences with local XAI
└── README.md
```

## Requirements

- Python 3.x
- scikit-learn (`sklearn`)
- tensorflow
- scipy
- matplotlib
- pandas
- numpy
- joblib



