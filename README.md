🍷 Wine Quality Prediction

📌 Project Overview

The Wine Quality Prediction project aims to predict the quality of wine based on its chemical properties. Using a dataset containing various physicochemical tests (such as acidity, sugar content, pH, and alcohol percentage), we train machine learning models to classify and regress wine quality.

This project serves as a practical implementation for both regression and classification tasks in machine learning, using Random Forest and XGBoost algorithms.

🎯 Objectives

Predict wine quality score (Regression).

Classify wine into quality categories (Classification).

Practice data preprocessing, EDA, feature engineering, and model building.

Compare performance of Random Forest and XGBoost.

📂 Dataset

File: WineQT.csv

Source: Public wine quality dataset.

Features:

fixed acidity

volatile acidity

citric acid

residual sugar

chlorides

free sulfur dioxide

total sulfur dioxide

density

pH

sulphates

alcohol

quality (Target variable)

🛠️ Technologies Used

Python 🐍

Pandas, NumPy – Data handling

Matplotlib, Seaborn – Visualization

Scikit-learn – ML models and metrics

XGBoost – Advanced boosting model (optional)

📊 Workflow

Import Libraries & Dataset

Data Exploration & Cleaning

Exploratory Data Analysis (EDA)

Feature Selection & Scaling

Model Training – Random Forest & XGBoost

Evaluation – MAE, RMSE (Regression) & Accuracy, Confusion Matrix (Classification)

Model Saving for future predictions

🚀 How to Run

Clone this repository:

Bash
git clone https://github.com/yourusername/wine-quality-prediction.git
cd wine-quality-prediction
Install dependencies:

Bash
pip install -r requirements.txt
Run the Jupyter Notebook:

Bash
jupyter notebook Wine_Quality_Prediction.ipynb
For prediction using a saved model:

Python
import pickle
model = pickle.load(open("wine_quality_model.pkl", "rb"))
sample = [[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
prediction = model.predict(sample)
print("Predicted Quality:", prediction)
📈 Results

Model Type	Metric	Result
Regression	RMSE	0.546
Classification	Accuracy	0.917
💾 Model Saving

The trained Random Forest model is saved as:

Bash
wine_quality_model.pkl
