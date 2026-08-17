# Breast Cancer Classification Lab

## a. Problem Statement

The goal of this project is to compare multiple supervised machine learning classification models on a public medical diagnosis dataset and deploy the comparison as an interactive Streamlit application. The app allows a user to upload test data, select a trained model, and view evaluation metrics, a confusion matrix, and a classification report.

## b. Dataset Description

Dataset used: **UCI Breast Cancer Wisconsin Diagnostic dataset**.

This is a binary classification dataset where each record describes measurements computed from a digitized image of a breast mass. The target variable indicates whether the tumor is malignant or benign.

- Source: UCI Machine Learning Repository, accessed through `sklearn.datasets.load_breast_cancer`
- Number of instances: 569
- Number of features: 30 numeric features
- Target column: `target`
- Target classes: `0 = malignant`, `1 = benign`

This dataset satisfies the assignment constraints because it has more than 500 instances and more than 12 features.

## c. GitHub Repository Link

Add your GitHub repository link here:

`<PASTE_GITHUB_REPOSITORY_LINK>`

## Live Streamlit App Link

Add your deployed Streamlit app link here:

`<PASTE_STREAMLIT_APP_LINK>`

## d. Models Used

The assignment text says six models must be implemented, while the list/table contains five. This project implements the five listed models and adds Support Vector Machine as a sixth model.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9386 | 0.9468 | 0.9577 | 0.9444 | 0.9510 | 0.8689 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| Support Vector Machine | 0.9825 | 0.9967 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Usually performs strongly on this dataset because the numeric cell-measurement features are well suited to a linear decision boundary after scaling. |
| Decision Tree | Easy to interpret, but it can overfit if the tree becomes too deep. A limited depth is used to improve generalization. |
| KNN | Performs well when similar cases are close in feature space, but it is sensitive to feature scaling and the value of `k`. |
| Naive Bayes | Fast and simple, but its independence assumption may be too strong because many tumor measurement features are correlated. |
| Random Forest | Usually robust because it combines multiple trees and reduces the instability of a single decision tree. |
| Support Vector Machine | Often performs strongly on medium-sized numeric datasets after scaling, especially with an RBF kernel. |
| Overall Winner | Support Vector Machine and Logistic Regression both achieve the best F1 and MCC, but Support Vector Machine has the highest AUC. Therefore, SVM is selected as the overall winner for this split. |

## How To Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models and generate CSV files:

```bash
python model/train_models.py
```

3. Start Streamlit:

```bash
streamlit run app.py
```

## Repository Structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- breast_cancer_uci_full.csv
|-- metrics_summary.csv
|-- metadata.json
|-- model/
    |-- train_models.py
    |-- *.joblib
```

## Streamlit App Features

- CSV upload option for test data
- Model selection dropdown
- Accuracy, AUC, precision, recall, F1, and MCC metrics
- Confusion matrix
- Classification report
- All-model comparison table
