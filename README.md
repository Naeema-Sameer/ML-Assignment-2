# PhishGuard AI – Machine Learning Based Phishing Website Detection

## 1. Problem Statement

Phishing websites imitate legitimate websites to deceive users into sharing sensitive information such as login credentials and financial details.

The objective of this project is to develop and compare five machine learning classification models to distinguish phishing websites from legitimate websites using URL and website-related features.

## 2. Dataset Description

**Dataset Name:** PhiUSIIL Phishing URL Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/967/phiusiil_phishing_url_dataset

| Attribute | Description |
|---|---|
| Dataset Records | 552 |
| Usable Labelled Records | 551 |
| Input Features | 50 numerical features |
| Problem Type | Binary Classification |
| Target Variable | label |
| Classes | Legitimate and Phishing |

One record with a missing target label was excluded during preprocessing.

The original dataset labels were converted for model training as follows:
- 0 – Legitimate
- 1 – Phishing

## 3. GitHub Repository

**Repository Link:**  
https://github.com/Naeema-Sameer/ML-Assignment-2

## 4. Model Performance Comparison

Five machine learning classification models were trained and evaluated using the same dataset and test split.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9928 | 1.0000 | 1.0000 | 0.9787 | 0.9892 | 0.9839 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 0.9928 | 0.9890 | 1.0000 | 0.9787 | 0.9892 | 0.9839 |
| Gaussian Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM | 0.9928 | 1.0000 | 1.0000 | 0.9787 | 0.9892 | 0.9839 |

## 5. Model Observations

- **Logistic Regression:** Achieved 99.28% accuracy and perfect precision. Its recall was 97.87%, indicating that a small number of phishing websites were missed.
- **Decision Tree:** Achieved 100% across all six evaluation metrics, correctly classifying every test instance.
- **KNN:** Achieved 99.28% accuracy with perfect precision and 97.87% recall. It demonstrated strong classification performance.
- **Gaussian Naive Bayes:** Achieved 100% across all six evaluation metrics, correctly classifying every test instance.
- **SVM:** Achieved 99.28% accuracy, perfect precision and 97.87% recall, demonstrating strong classification performance.

## 6. Overall Best Model

Decision Tree and Gaussian Naive Bayes were the best-performing models in this experiment. Both achieved 100% across all six evaluation metrics on the test dataset.

However, these results are based on a relatively small test set of 138 records. Further evaluation using a larger, independent dataset is recommended to assess their generalisation performance.
