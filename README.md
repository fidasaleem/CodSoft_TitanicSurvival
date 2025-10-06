# Titanic Survival Prediction

This repository contains a machine learning project to predict survival on the Titanic using passenger data. The project demonstrates data preprocessing, exploratory data analysis (EDA), feature engineering, and model building to predict whether a passenger survived or not.

---

## 📂 Project Structure

Titanic_Project/
├── Titanic_Survival.ipynb ← Jupyter Notebook with EDA and modeling
├── titanic.py ← Streamlit app  for prediction
├── Requirements.txt ← Python dependencies
├── titanic_rf_model.pkl ← Saved trained model
└── titanic_rf_threshold.pkl


---

## 💻 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/fidasaleem/CodSoft_TitanicSurvival_.git
cd Titanic_Project
```
### 2. Install dependencies
```bash
pip install -r Requirements.txt
```
### 3.Run the Streamlit app 
```bash
streamlit run titanic.py
```

## Project Highlights

- Exploratory Data Analysis (EDA) of Titanic dataset
- Data preprocessing and handling missing values
- Feature engineering to improve model performance
- Model training using algorithms like Logistic Regression, Random Forest, and Gradient Boosting
- Evaluation using Accuracy, F1-score, and Confusion Matrix
- Interactive prediction via Streamlit app
