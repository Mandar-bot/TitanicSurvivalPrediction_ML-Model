# 🚢 Titanic Survival Prediction: Logistic Regression & Naive Bayes

## Project Overview
This project focuses on predicting passenger survival on the Titanic using machine learning. We employ a standard data science workflow, including rigorous data cleaning, feature engineering, and training two fundamental classification models: **Logistic Regression** and **Gaussian Naive Bayes**.

## 💾 Dataset
The analysis uses the classic **Titanic training dataset** (`train.csv`), which contains various passenger attributes (e.g., Pclass, Sex, Age, Fare) and the survival outcome (0 or 1).

## 📊 Key Steps & Notebook Structure

The `Logistic Regression and Naive Bayes for Titanic Survivor Prediction.ipynb` notebook details the following steps:

1.  **Data Loading & Exploration:** Initial loading of `train.csv` and visualization of survival counts.
2.  **Data Preprocessing (Cleaning & Imputation):**
    * Missing **Age** values are imputed based on the passenger's gender (male/female averages).
    * Missing **Embarked** values are filled with the mode.
    * Sparse columns like **`Cabin`**, `Name`, `Ticket`, and `PassengerId` are dropped to prevent non-numeric errors during modeling.
3.  **Feature Engineering:** Categorical features (`Sex`, `Embarked`) are converted into numerical format using **One-Hot Encoding** (`pd.get_dummies`).
4.  **Model Training & Evaluation:**
    * The data is split and features are **scaled** using `StandardScaler`.
    * Both **Logistic Regression** and **Gaussian Naive Bayes** models are trained and their performance is evaluated using **Accuracy**, **Confusion Matrix**, and a **Classification Report**.

## 🛠️ Requirements
To run this notebook, you need the following Python libraries:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn` (`sklearn`)

You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
