# Credit Card Fraud Detection System 💳

## Project Overview
This project is an end-to-end machine learning system designed to detect
fraudulent credit card transactions. It covers the complete workflow from
data analysis and model training to saving the trained model and using it
in a user-facing application.

## Problem Statement
Credit card fraud is a major challenge in financial systems. The goal of
this project is to build a machine learning model that can classify a
transaction as fraudulent or legitimate based on anonymized transaction
features.

## Project Files
- analysis_model.ipynb 
  This notebook performs data analysis, preprocessing, model training,
  and evaluation. It also saves the trained machine learning pipeline
  as a serialized file.

- `fraud_detection_pipeline.pkl`  
  This file contains the trained machine learning pipeline generated
  from the analysis notebook. It is used directly for making predictions.

- `fraud_detection.py`  
  A Streamlit-based application that loads the trained pipeline and allows
  users to input transaction details to check whether a transaction is
  fraudulent or valid.

## Machine Learning Approach
- Supervised classification
- Feature-based learning using anonymized transaction features (V1–V28)
  along with the transaction Amount
- Pipeline-based preprocessing and model usage

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## How the System Works
1. Transaction data is analyzed and used to train a machine learning model
2. The trained model and preprocessing steps are saved as a pipeline
3. The saved pipeline is loaded in a Streamlit application
4. User inputs transaction details
5. The system predicts whether the transaction is fraudulent and shows
   the probability score

## Learning Outcomes
- Understanding of credit card fraud detection using machine learning
- Experience with building and saving ML pipelines
- Practical exposure to model reuse and prediction
- Deploying machine learning logic through a simple application interface

## Author
Om Shete
