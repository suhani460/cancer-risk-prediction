# 🩺 Cancer Risk Level Prediction

An interactive Machine Learning web application that predicts cancer risk levels based on multiple lifestyle, environmental, genetic, and health-related risk factors.

The application is built using **Python, Scikit-learn, Streamlit, and SMOTE** to demonstrate a complete machine learning workflow with an interactive dashboard.

## 🚀 Live Demo

🔗 **Streamlit App:**https://studentmanagementsystem-solydndrnbs6zjw37ehidd.streamlit.app/

## 📌 Project Overview

Cancer is influenced by several factors, including age, lifestyle habits, family history, environmental exposure, and genetic conditions.

This project uses Machine Learning to analyze these risk factors and classify individuals into different cancer risk categories based on patterns learned from the dataset.

The application provides an interactive interface where users can explore the dataset, enter risk-factor information, view predictions, and evaluate model performance.

**Note:** This project is developed for educational and research purposes only. It is not a medically validated diagnostic or clinical decision-making tool.

## ✨ Features

- 📊 Interactive dashboard for dataset visualization.
- 🧬 Cancer risk level prediction based on user inputs.
- 📈 Dataset exploration and statistical insights.
- 🤖 Random Forest classification model.
- ⚖️ SMOTE for handling class imbalance.
- 📉 Model evaluation using accuracy, F1-score, classification report, and confusion matrix.
- 📥 Downloadable dataset.
- 🌐 Interactive web application built with Streamlit.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| Imbalanced-learn | SMOTE |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Streamlit | Web Application |

## 📂 Project Structure

```text
Cancer-Risk-Level-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── cancer-risk-factors.csv
│
└── notebooks/
    └── Cancer_Risk_Prediction.ipynb
```

*Note: Adjust the folder structure above to match the actual files and folders in your repository.*

## 📊 Dataset Information

The project uses the **Cancer Risk Factors & Types** dataset.

- Dataset Size: 2,000 records
- Features: Demographic, lifestyle, genetic, environmental, and health-related factors.
- Target Variable: `Risk_Level`

Dataset Source:

- [Kaggle Dataset](https://www.kaggle.com/datasets/tarekmasryo/cancer-risk-factors-dataset)
- [GitHub Dataset Repository](https://github.com/tarekmasryo/cancer-risk-factors-data)

## ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Loading and Exploration
3. Data Preprocessing
4. Feature Selection
5. Train-Test Split
6. Handling Class Imbalance using SMOTE
7. Model Training using Random Forest Classifier
8. Model Evaluation
9. Streamlit Application Development
10. Cloud Deployment

### 🤖 Model Used
