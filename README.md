# JPMC-Loan-Risk-FICO-Quantization-


#  Loan Default Prediction & FICO Score Risk Binning

This repo contains tools for credit risk modeling, focused on predicting loan defaults and transforming FICO scores for machine learning.

## 📁 Contents

### 3️⃣ Loan Default Risk Model
- Input: borrower features (income, FICO, debt, employment)
- Output: Probability of Default (PD) + Expected Loss
- Model: Logistic Regression (AUC ~0.999)
- Python function: `predict_expected_loss(borrower_dict)`

### 4️⃣ FICO Score Quantization (Rating Buckets)
- Converts continuous FICO scores (300–850) into categorical buckets
- Optimized using **greedy log-likelihood maximization**
- Python function: `map_fico_to_rating(score, bucket_bounds)`

## ⚙️ Techniques Used
- Logistic regression, AUC evaluation
- Greedy recursive splitting (log-likelihood based)
- Quantization / binning for categorical ML pipelines

## 🏦 Applications
- Retail banking loan books
- Loss allowance modeling
- Feature engineering for credit risk models

---
