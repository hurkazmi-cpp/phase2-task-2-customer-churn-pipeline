# Customer Churn Prediction Pipeline

An end-to-end, production-ready machine learning pipeline developed to predict telecommunications customer churn using the **Scikit-learn Pipeline API**. 

The project automates data cleaning, scaling, hyperparameter tuning, and model serialization into a seamless, leak-proof workflow.

## 🚀 Key Features
* **Automated Preprocessing:** Utilizes `ColumnTransformer` to handle missing values, scale continuous features, and encode categorical data without data leakage.
* **Model Optimization:** Automates hyperparameter search using `GridSearchCV`.
* **Dual-Model Comparison:** Evaluates and compares **Logistic Regression** and **Random Forest** models.
* **Production Deployment:** Serializes the final trained model into a production-ready `.pkl` asset using `joblib`.

## 🛠️ Tech Stack
* Python 3
* Pandas & NumPy
* Scikit-Learn
* Matplotlib & Seaborn
* Joblib

## 📊 Dataset
The pipeline uses the **Telco Customer Churn dataset** (~7,043 rows, 20 features), covering:
* **Demographics:** Gender, SeniorCitizen, Partner, Dependents.
* **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport, etc.
* **Account Info:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.

## 🧬 Pipeline Architecture
1. **Data Cleaning:** Handles empty strings, fixes data types, and maps target labels.
2. **Numerical Pipeline:** Missing value imputation (`median`) $\rightarrow$ Feature scaling (`StandardScaler`).
3. **Categorical Pipeline:** Missing value imputation (`most_frequent`) $\rightarrow$ Encoding (`OneHotEncoder`).
4. **Model Tuning:** Cross-validated grid search to find the best estimator.
5. **Serialization:** Dumps the pipeline architecture into `telco_churn_production_pipeline.pkl`.

## 💻 Getting Started

### Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
