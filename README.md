# Loan Approval Prediction System

ML-based loan approval prediction system using scikit-learn. Includes data cleaning, EDA with visualizations, feature encoding (Label & OneHot), and correlation analysis. Compares Logistic Regression, KNN, and Naive Bayes models. Achieves 87% accuracy with a reusable inference pipeline exported via joblib for deployment.

## Dataset

- **Source:** `loan_approval_data.csv`
- **Samples:** 1,000 applicants
- **Features:** 19 input features including income, credit score, DTI ratio, employment status, loan amount, and more
- **Target:** `Loan_Approved` (Yes / No)

## Project Workflow

### 1. Data Cleaning
- Handled missing values (~50 per column) using scikit-learn's `SimpleImputer`
- **Numerical features** filled with mean
- **Categorical features** filled with mode

### 2. Exploratory Data Analysis (EDA)
- Class distribution (pie chart)
- Gender and education level analysis (bar plots)
- Income distribution (histogram)
- Outlier detection (box plots)
- Credit score vs. loan approval analysis

### 3. Feature Encoding
- **Label Encoding** for ordinal features (`Education_Level`, `Loan_Approved`)
- **One-Hot Encoding** for nominal features (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`)

### 4. Correlation Analysis
- Heatmap of all numerical features
- Top positive correlators: `Credit_Score` (+0.45), `Applicant_Income` (+0.12)
- Top negative correlators: `DTI_Ratio` (-0.44), `Loan_Amount` (-0.13)

### 5. Feature Engineering
- Squared terms: `DTI_Ratio_sq`, `Credit_Score_sq`
- Log transform: `Applicant_Income_log` (to handle skewness)

### 6. Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 87.0%    | 77.8%     | 80.3%  | 79.0%    |
| KNN (k=7)            | 74.5%    | 60.9%     | 45.9%  | 52.3%    |
| Naive Bayes          | 86.5%    | 80.4%     | 73.8%  | 76.9%    |

**Best model:** Logistic Regression (after feature engineering) — highest accuracy, recall, and F1 score.

### 7. Pipeline Export
- Built a reusable `sklearn.pipeline.Pipeline` (StandardScaler + LogisticRegression)
- Exported as `loan_pipeline.pkl` using joblib for deployment

## Usage

```python
import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load('loan_pipeline.pkl')

# Predict on new data (must match the 27 engineered features)
prediction = pipeline.predict(new_data)
```

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- joblib
