# Cloud AI for Credit Card Fraud Detection

An XGBoost fraud-detection model trained on a heavily imbalanced dataset, plus a proposed design for running it in real time on AWS.

Built as group coursework for **COS7054-B Cloud AI** (MSc Artificial Intelligence and Machine Learning, University of Bradford, November 2026).

## Results

On the held-out test set (the newest 20% of transactions: 56,746 transactions, 74 of them fraud):

| Metric | Tuned XGBoost (ours) | Replication of published SMOTE-ENN setup |
|---|---|---|
| Precision | **0.849** | 0.458 |
| Recall | **0.757** | 0.743 |
| F1-score | **0.800** | 0.567 |
| ROC-AUC | **0.988** | 0.928 |
| PR-AUC | **0.805** | 0.348 |
| Frauds detected (of 74) | **56** | 55 |
| False positives | **10** | 65 |

Scoring a single transaction took about **7.7 ms** in Google Colab.

## Dataset

[ULB Credit Card Fraud dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 European card transactions, 492 of them fraud (0.172%). The features are `Time`, `Amount` and 28 anonymised PCA components (`V1`–`V28`). The notebook downloads the data automatically with `kagglehub`.

## Approach

1. **Data quality check:** no missing values and no negative amounts, but 1,081 duplicate rows. These were removed so the same transaction can't appear in both training and test data.
2. **Time-aware split:** transactions were sorted by time and split 70% train / 10% validation / 20% test, so the model is always tested on transactions newer than the ones it was trained on.
3. **Feature scaling:** `Amount` and hour of day were standardised, with the scaler fitted on training data only. Raw `Time` was dropped.
4. **Class imbalance:** we compared SMOTE, SMOTE-ENN and undersampling against class weighting. Every resampling method lowered validation PR-AUC (no resampling: 0.849, SMOTE: 0.795, SMOTE-ENN: 0.793), so the final model uses `scale_pos_weight` instead.
5. **Hyperparameter tuning:** Optuna, 100 trials, five-fold time-series cross-validation, with PR-AUC as the objective.
6. **Decision threshold:** set to 0.434, chosen by maximising F1 on out-of-fold predictions rather than on the test set.
7. **Comparison with a published paper:** we re-ran a published XGBoost + SMOTE-ENN setup under the same leakage-free split. It reached a PR-AUC of only 0.348, well below what the paper reported.

## Proposed AWS architecture

The design has two paths:

- **Real-time scoring:** Amazon Kinesis receives transactions, AWS Lambda preprocesses them, and a SageMaker endpoint scores each one before authorisation completes.
- **Retraining:** SageMaker Processing and Training Jobs retrain the model on new data stored in S3. SageMaker Model Monitor watches for drift and starts retraining when needed. CloudWatch keeps logs and an audit trail.

This architecture was designed but **not deployed**. The model was trained and evaluated in Google Colab.

## Repository contents

| File | Description |
|---|---|
| `cloudai_CW1__finalipynb.ipynb` | Full pipeline: data audit, preprocessing, resampling comparison, paper replication, Optuna tuning, final evaluation, latency test and model export |

## Running it

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run all cells. Dependencies are installed in the first cell (`xgboost`, `optuna`, `imbalanced-learn`, `scikit-learn`, `shap`, `kagglehub`).
3. Trained models are saved to Google Drive. Change `SAVE_DIR` if you want them somewhere else.

## Tech stack

Python · XGBoost · Optuna · scikit-learn · imbalanced-learn · pandas · NumPy · Matplotlib · Google Colab · AWS (architecture design)

## Authors

- Edina Bajric
- Alnura Maxyutova
