# QoE Prediction from QoS Parameters in Cameroon Telecommunication Networks

This document outlines the step-by-step implementation plan for building a research-grade machine learning project predicting Quality of Experience (QoE) from Quality of Service (QoS) parameters, and the accompanying interactive Streamlit dashboard.

## User Review Required

Please review the proposed approach. Are there any specific libraries (like a particular version of pandas or scikit-learn) you prefer, or any specific constraints on the environment? If this plan looks good, I will proceed with creating the notebook and dashboard.

## Open Questions

- Do you have a specific color palette preference for the dashboard to match any institutional/university branding?
- Should the `ydata-profiling` EDA report be generated inline in the notebook, or saved as a separate HTML file? (The plan defaults to inline).

## Proposed Changes

We will divide the work into two main components: the Jupyter Notebook for research and modeling, and the Streamlit application for the interactive dashboard.

### Jupyter Notebook (`qoe_prediction.ipynb`)

#### [NEW] [qoe_prediction.ipynb](file:///c:/Users/USER/Desktop/QoE%20Prediction%20Model/qoe_prediction.ipynb)
We will create this notebook and structure it exactly according to the requested 20 sections:

1.  **Project Initialization**: Installing and importing libraries (pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, shap, etc.).
2.  **Load and Analyze Dataset**: Comprehensive EDA, statistical summaries, correlation analysis, and visualizations of `Dataset_QoE_Cameroon.csv`.
3.  **Automatic Feature Classification**: Grouping features logically (QoS, QoE, Contextual, Behavioral).
4.  **Target Variable Analysis**: Distribution and statistical analysis of `QoE_Score`.
5.  **Data Cleaning**: Handling missing values, outliers, malformed strings, and data types.
6.  **Feature Engineering**: Creating composite scores and telecommunication-specific metrics.
7.  **Encoding and Scaling**: Applying `OneHotEncoder`, `LabelEncoder`, and `StandardScaler`/`RobustScaler` within pipelines.
8.  **Train-Test Split**: 80/20 split preventing data leakage.
9.  **Baseline Modeling**: Linear Regression, Ridge, and Lasso.
10. **Advanced Machine Learning Models**: Tree-based models, ensemble methods, and SVR.
11. **Hyperparameter Tuning**: Optimizing top performers (RF, XGBoost, CatBoost, LightGBM) using `RandomizedSearchCV`/`GridSearchCV`.
12. **Model Evaluation**: Comprehensive metrics (RMSE, MAE, R²) and residual analysis to select the BEST model.
13. **Explainable AI (SHAP)**: Global and local explainability using SHAP summary, importance, dependence, and waterfall plots.
14. **Telecom Engineering Interpretation**: Analyzing SHAP outputs from a network engineer's perspective.
15. **Optional Classification System**: Discretizing `QoE_Score` (Poor, Fair, Good, Excellent) and training classifiers.
16. **Model Saving**: Exporting the best model, scalers, and encoders via `joblib`.
17. *(Placeholder for Streamlit - refer to app.py below)*
18. *(Placeholder for Advanced Features - refer to app.py below)*
19. **Final Research Conclusions**: Summary of findings, telecom insights, and future work.
20. **Notebook Quality**: Ensuring markdown, comments, and structure are research-grade.

### Streamlit Dashboard (`app.py`)

#### [NEW] [app.py](file:///c:/Users/USER/Desktop/QoE%20Prediction%20Model/app.py)
We will create a multi-page, interactive Streamlit application acting as the interface for the trained models.

-   **Sidebar / Navigation**: Tabs for EDA, Modeling, Explainability, and Scenario Simulator.
-   **Telecom KPI Input Controls**: Sliders for latency, jitter, packet loss, bandwidth, etc.
-   **Real-Time QoE Prediction**: Instant prediction feedback based on slider inputs.
-   **Algorithm Selector**: Dropdown to swap between the trained RF, XGBoost, CatBoost, and LightGBM models.
-   **Interactive SHAP**: Displaying a force/waterfall plot for the current input configuration to explain *why* the prediction was made.
-   **Visualization & Scenario Simulator**: Interactive plots to understand how modifying specific KPIs degrades or improves QoE.
-   **Dataset Upload**: Allowing users to upload a new CSV to generate batch predictions.

## Verification Plan

### Automated Verification
- Ensure the Jupyter Notebook executes from top to bottom without errors.
- Verify the saved model files (`best_model.pkl` or `.joblib`) are successfully created and loadable.
- Run `streamlit run app.py` and verify the dashboard starts without errors.

### Manual Verification
- Review the markdown and explanations in the notebook to ensure they meet the standard of a Master's level research project and use proper telecom engineering terminology.
- Manually test the Streamlit dashboard: interact with sliders to simulate network degradation (e.g., high latency, high packet loss) and verify the QoE score reacts logically and the SHAP plot updates appropriately.
