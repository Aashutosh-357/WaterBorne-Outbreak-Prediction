# 🛡️ AquaShield AI – Waterborne Disease Outbreak Prediction

> **AI-powered system trained on 5.25 million records to predict waterborne disease outbreaks across all states and districts of India.**

---

## 📋 Project Overview

AquaShield AI is a capstone project that uses **XGBoost (Gradient Boosting)** to predict the risk of waterborne disease outbreaks based on water quality, sanitation infrastructure, and environmental conditions. The system is designed to assist public health officials in proactive decision-making.

### Key Highlights
- 🗂️ **Dataset**: 5.25 million records covering all of India
- 🤖 **Model**: XGBoost Classifier (histogram-based for big data)
- 🎯 **Accuracy**: 88.44% on the test set
- 🖥️ **Dashboard**: Interactive Streamlit web application
- 📊 **Features**: 9 input parameters for prediction

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AIOPS_Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the Dataset
Download the 1.2GB original dataset from [Google Drive](https://drive.google.com/drive/folders/17N-kTj4xFRHqr8d83iUW8OmAXObftE0O?usp=sharing) and place it at:
```
data/raw/waterborne_disease.csv
```

### 5. Train the Model
```bash
python src/model_trainer.py
```

### 6. Launch the Dashboard
```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
AIOPS_Project/
│
├── data/                        # Never upload to GitHub (Too large!)
│   ├── raw/                     # The 1.2GB dataset
│   └── processed/               # Cleaned data for the Streamlit app
│
├── models/                      # Saved AI models
│   └── xgboost_model.pkl        # The trained brain of the app
│
├── notebooks/                   # Jupyter / Colab notebooks
│   └── 01_data_training.ipynb   # Training experiments
│
├── src/                         # Source code (Helper modules)
│   ├── __init__.py
│   ├── data_loader.py           # Data loading & preprocessing
│   └── model_trainer.py         # Model training & evaluation
│
├── app/                         # Streamlit Web App
│   ├── app.py                   # Main dashboard
│   ├── components.py            # Charts & visualizations
│   └── assets/                  # Images, logos
│
├── reports/                     # Faculty submissions
│   ├── Capstone_Report.pdf
│   └── figures/                 # SHAP graphs & maps
│
├── .gitignore                   # Ignores data & model files
├── requirements.txt             # Python dependencies
├── main.py                      # Standalone training script
└── README.md                    # This file!
```

---

## 🤖 Model Details

| Parameter       | Value           |
|----------------|-----------------|
| Algorithm       | XGBClassifier   |
| n_estimators    | 100             |
| learning_rate   | 0.1             |
| max_depth       | 6               |
| tree_method     | hist            |
| eval_metric     | logloss         |

### Input Features
| # | Feature              | Description                       |
|---|----------------------|-----------------------------------|
| 1 | `is_urban`           | Urban (1) or Rural (0)            |
| 2 | `population_density` | People per km²                    |
| 3 | `water_source`       | Piped, Borewell, River, etc.      |
| 4 | `water_treatment`    | Chlorination, Filtration, etc.    |
| 5 | `ph`                 | Water pH level                    |
| 6 | `avg_temperature_c`  | Average temperature (°C)          |
| 7 | `avg_rainfall_mm`    | Average rainfall (mm)             |
| 8 | `avg_humidity_pct`   | Average humidity (%)              |
| 9 | `flooding`           | Flooding present (1) or not (0)   |

---

## 📊 Performance

```
              precision    recall  f1-score   support

    Safe (0)       0.86      0.86      0.86    419,666
Outbreak (1)       0.90      0.90      0.90    630,334

    accuracy                           0.88  1,050,000
   macro avg       0.88      0.88      0.88  1,050,000
weighted avg       0.88      0.88      0.88  1,050,000
```

---

## 🛠️ Tech Stack

- **Python 3.12**
- **XGBoost** – Gradient Boosting for classification
- **Pandas & NumPy** – Data manipulation
- **Scikit-learn** – Preprocessing & evaluation
- **Streamlit** – Interactive web dashboard
- **Plotly** – Beautiful interactive charts
- **Joblib** – Model serialization

---

## 👤 Author

**Ashu** – Capstone Project

---

## 📝 License

This project is for educational purposes as part of the Capstone submission.
