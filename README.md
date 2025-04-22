# 🔍 Agent Churn Prediction Model

A machine learning project designed to predict agent churn in an insurance context with **94% accuracy**. This project uses **realistic business datasets**, performs **feature engineering**, and applies **supervised learning models** to generate actionable insights.

---

## 📌 Problem Statement

Churn prediction is crucial for retaining valuable agents. The goal is to develop a model that can predict which agents are most likely to leave based on historical and behavioral data.

---

## 📊 Dataset

The model uses three core datasets:

- `Agent_Master_Data.csv`
- `New_Business_Data.csv`
- `Renewal_Data.csv`

These datasets contain agent activity, new policy sales, and renewal histories.

---

## 🧠 Approach

1. **Data Preprocessing**
   - Merged multi-source datasets
   - Handled null values, encoding, and outliers
2. **Feature Engineering**
   - Extracted key metrics: renewal ratios, inactivity periods, sales frequency, etc.
3. **Modeling**
   - Trained with models like **Logistic Regression**, **Random Forest**, and **XGBoost**
   - Achieved **94% accuracy** with XGBoost
4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score
   - Visualizations: Confusion Matrix, ROC Curve

---

## ⚙️ Tech Stack

- **Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Tools**: Jupyter Notebook, VS Code

---

## 📈 Results

- Final model (XGBoost) achieved **94% accuracy**
- Feature importance plots provided business insights
- Early prediction helps in proactive agent retention

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/kanishka-raisania/AgentChurnPrediction.git
   cd AgentChurnPrediction
