# QoE Prediction from QoS Parameters in Cameroon Telecommunication Networks

![University of Buea](https://upload.wikimedia.org/wikipedia/en/thumb/e/ef/University_of_Buea_logo.png/150px-University_of_Buea_logo.png)

**Master of Science in Telecommunications and Network Engineering Project**  
*University of Buea, Cameroon*

---

## 📌 Project Overview
This research-grade machine learning project aims to predict **Quality of Experience (QoE)** based on **Quality of Service (QoS)** key performance indicators (KPIs) collected from telecommunication networks in Cameroon.

By moving beyond simple network metrics (like latency and packet loss) and predicting actual perceived user experience, this project helps network engineers prioritize traffic, diagnose bottlenecks, and optimize resource allocation effectively.

The repository includes:
1. **A highly documented Jupyter Notebook** covering Exploratory Data Analysis (EDA), feature engineering, model training, and Explainable AI (SHAP).
2. **An Interactive Streamlit Dashboard** allowing real-time scenario simulation and batch predictions for network administrators.

---

## 🚀 Features
- **Data Preprocessing Pipeline:** Robust handling of malformed telecom data, missing values, and outliers.
- **Feature Engineering:** Advanced telecom KPIs (e.g., `Throughput_Ratio`, `Network_Instability_Index`).
- **Machine Learning Models:** Comparisons across Linear models, Random Forest, XGBoost, CatBoost, LightGBM, and SVR.
- **Explainable AI (SHAP):** Transparent predictions showing exactly how Jitter, Latency, and Packet Loss affect a user's MOS (Mean Opinion Score).
- **Streamlit Dashboard:** A production-ready UI for real-time QoE simulation.

---

## ⚙️ Installation Guide

### Prerequisites
Make sure you have **Python 3.10+** installed on your machine.

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/QoePredictModel.git
cd QoePredictModel
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
**On Windows:**
```powershell
python -m venv qoepredict
.\qoepredict\Scripts\activate
```
**On macOS/Linux:**
```bash
python3 -m venv qoepredict
source qoepredict/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm matplotlib seaborn plotly shap streamlit jupyter joblib
```

---

## 📖 How to Run the Project

### Part 1: Running the Jupyter Notebook (Model Training)
You must run the notebook first to train the machine learning models and generate the required `.pkl` files for the dashboard.
1. Start the Jupyter server:
   ```bash
   jupyter notebook
   ```
2. Open `qoe_prediction.ipynb` in your browser.
3. Click **Cell > Run All** (or **Run > Run All Cells**).
4. Verify that `best_model.pkl` and `preprocessor.pkl` have been generated in your project folder.

### Part 2: Running the Interactive Dashboard
Once the models are saved, you can launch the Streamlit dashboard:
```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`) where you can interact with the QoE Scenario Simulator.

---

## 📊 Repository Structure
```text
QoePredictModel/
│
├── Dataset_QoE_Cameroon.csv   # Raw dataset containing QoS and QoE variables
├── create_notebook.py         # Script to programmatically generate the Jupyter Notebook
├── qoe_prediction.ipynb       # Main research notebook (EDA, Training, SHAP)
├── app.py                     # Streamlit Interactive Dashboard
├── best_model.pkl             # Trained model (Generated after running the notebook)
├── preprocessor.pkl           # Scikit-learn Pipeline (Generated after running the notebook)
└── README.md                  # Project documentation
```

---

## 🎓 Academic Context
This work is conducted as part of a Master's degree requirement in Telecommunications and Network Engineering. It bridges the gap between raw network statistics (QoS) and human perception (QoE), providing actionable insights for network operators in Cameroon to improve service quality.
