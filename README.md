# Insurance-Premium-Predictor

This is a kaggle competetion propject. The project predicts the **insurance premium amount** based on customer details using various machine learning models. The best model is deployed as a **Streamlit web app**. 

## Aim

The goal is to build a regression model that predicts the **premium amount** a customer would likely pay, based on multiple demographic and behavioral features.

## Project Overview

- Dataset: Real-world insurance dataset with demographic and policy features.
- Objective: Predict `Premium Amount` using features like age, income, health score, vehicle age, and more.
- Evaluation Metrics: RMSLE, RMSE, MAE, R² Score
- Model Logged with: [MLflow](https://mlflow.org/)
- UI Built With: [Streamlit](https://streamlit.io/)

## ML Models Used

- Linear Regression
- Decision Tree Regressor  
- Random Forest
- XGBoost 

**NOTE:** The best performing model based on the evaluation metrics was **XGBoost Regressor**, and it is deployed via Streamlit.

## Dataset Info

- **Train File:** `train.csv` (~1.2 million records, used for training)
- **Test File:** `test.csv` (used for manual testing through Streamlit)
- **Sample Submission:** `sample_submission.csv` (Kaggle-style format for predictions)

**Features include:**

- Age, Gender, Annual Income
- Marital Status, Number of Dependents
- Education Level, Occupation
- Health Score, Smoking Status, Exercise Frequency
- Vehicle Age, Credit Score, Insurance Duration
- Property Type, Policy Type, Feedback, etc.

## ML Workflow

1. **EDA** on the `train.csv` file.
2. **Feature Engineering:**
   - Handling missing values
   - Categorical encoding using `OneHotEncoder`
   - Feature scaling using `StandardScaler`
3. **Model Training using Pipelines**
   - Four different models tried
   - Tracked via `MLflow`
4. **Model Evaluation**
   - Metrics: RMSLE, RMSE, MAE, R²
5. **Model Logging**
   - Best model logged and registered via `MLflow`
6. **Streamlit App**
   - Loads the best model and takes user input for real-time predictions.
  
## My way of working

I performed all the steps in ML workflow both manually as well as automated the pipelines in the project workflow using Pipeline and ColumnTransformer from sklearn library. 

Here's how I structured my workflow:

**Manual Implementation First:**
I started by manually handling EDA, preprocessing, feature engineering, and model training. This gave me a clear understanding of how the data and algorithms behave.

**Then Automated with Pipelines:**
After validating the manual steps, I transitioned to using Scikit-learn Pipelines and ColumnTransformer for streamlined, reusable, and production-ready code.

**Cross-Validation & Evaluation:**
Cross-validation and metric tracking were incorporated to ensure consistency and robustness.

**MLflow Integration:**
All experiments (parameters, metrics, models) were tracked using MLflow to facilitate comparison and easy model versioning.

**Model Deployment with Streamlit**
The project is deployed as a Streamlit web application where users can input customer details and get a real-time insurance premium prediction. The final Streamlit app is deployed on Streamlit Cloud, Heroku, or AWS, making it accessible to users.

## Models Compared

| Model             | RMSLE | RMSE  | MAE   | R² Score |
|-------------------|-------|-------|-------|----------|
| Linear Regression | 1.17  | 863.34| 667.28| 0.00     |
| Decision Tree     | 1.15  | 848.96| 643.66| 0.04     |
| Random Forest     | 1.17  | 855.92| 660.87| 0.02     |
| XGBoost           | 1.15  | 847.06| 646.53| 0.04     |   

## Evaluation Metrics

- **RMSLE** (Root Mean Squared Log Error) → used for model comparison
- **RMSE**, **MAE**, **R² Score** → for interpretability and robustness

## How to Run Locally

> Python 3.10+ recommended

1. **Install dependencies**

```bash
pip install -r requirements.txt
```
2. **Run Streamlit app**

```bash
streamlit run app.py
 ```

3. **The app will open in your browser at:** http://localhost:8501

4. **Fill in customer details and hit Predict Premium**

## Deployed Streamlit App

**Link to the Web page:** https://insurance-premium-predictor-zqspzehfdujgzv56hj9qx8.streamlit.app/


