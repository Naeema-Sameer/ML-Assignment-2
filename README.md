# PhishGuard AI – Machine Learning Based Phishing Website Detection

## a. Problem Statement

Phishing websites are designed to imitate legitimate websites to deceive users into sharing sensitive information such as login credentials and financial details.

The objective of this project is to develop and compare machine learning classification models that can distinguish phishing websites from legitimate websites using URL and website-related features.

## b. Dataset Description

- **Dataset Name:** PhiUSIIL Phishing URL Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset Link:** https://archive.ics.uci.edu/dataset/967/phiusiil_phishing_url_dataset
- **Problem Type:** Binary Classification
- **Records in the local dataset:** 552
- **Usable labelled records:** 551
- **Numerical input features:** 50
- **Target column:** `label`

The local dataset contains one record with a missing target value, which was excluded before model training.

The original target labels are:
- `0` – Phishing
- `1` – Legitimate

For model training, the labels were encoded as:
- `1` – Phishing
- `0` – Legitimate

The selected dataset satisfies the assignment's minimum requirements of 500 instances and 12 features.

## c. GitHub Repository Link

**Repository:** https://github.com/Naeema-Sameer/ML-Assignment-2

## d. Models Used

Five classification algorithms were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine (SVM)

### Model Performance Comparison

The following table presents the evaluation results obtained on the test dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|---------:|----:|-----------:|------:|---------:|----:|
| Logistic Regression | 0.9928 | 1.0000 | 1.0000 | 0.9787 | 0.9892 | 0.9839 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 0.9928 | 0.9890 | 1.0000 | 0.9787 | 0.9892 | 0.9839 |
| Gaussian Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Support Vector Machine | 0.9928 | 1.0000 | 1.0000 | 0.9787 | 0.9892 | 0.9839 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved 99.28% accuracy with perfect precision and AUC. It demonstrated strong classification performance, although its recall was slightly lower than the best-performing models. |
| Decision Tree | Achieved 100% across all six evaluation metrics on the test dataset, demonstrating perfect classification performance on this test split. |
| KNN | Achieved 99.28% accuracy and strong precision and recall. Its AUC was slightly lower than that of the other high-performing models. |
| Gaussian Naive Bayes | Achieved 100% across all six evaluation metrics on the test dataset, making it one of the best-performing models in this experiment. |
| Support Vector Machine | Achieved 99.28% accuracy, perfect precision and an AUC of 1.00, showing strong classification performance. |

### Overall Winner for the Dataset

Decision Tree and Gaussian Naive Bayes are the joint best-performing models based on the evaluated test-set metrics. Both achieved 100% across all six metrics.

These results reflect performance on the selected test set and should not be interpreted as a guarantee of equivalent performance on larger or independent datasets.
