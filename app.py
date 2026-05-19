import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG & THEME (University of Buea) ---
# White, Blue, Green
st.set_page_config(
    page_title="QoE Prediction Dashboard | Univ. of Buea",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Univ. of Buea Colors (White, Blue: #003366, Green: #008000)
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
    }
    .css-1d391kg { 
        background-color: #F0F4F8; 
    }
    h1, h2, h3 {
        color: #003366 !important;
    }
    .stButton>button {
        background-color: #008000;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #006400;
        color: white;
    }
    div[data-baseweb="select"] > div {
        border-color: #003366;
    }
    .sidebar-text {
        color: #003366;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_models():
    preprocessor = None
    best_model = None
    feature_names = None
    
    if os.path.exists('preprocessor.pkl'):
        preprocessor = joblib.load('preprocessor.pkl')
    if os.path.exists('best_model.pkl'):
        best_model = joblib.load('best_model.pkl')
    if os.path.exists('feature_names.pkl'):
        feature_names = joblib.load('feature_names.pkl')
        
    return preprocessor, best_model, feature_names

preprocessor, best_model, feature_names = load_models()

@st.cache_data
def load_default_data():
    if os.path.exists('Dataset_QoE_Cameroon.csv'):
        df = pd.read_csv('Dataset_QoE_Cameroon.csv', on_bad_lines='skip', encoding='utf-8', engine='python')
        df.columns = df.columns.str.strip()
        # Clean malformed numeric columns (same as notebook Section 5)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].astype(str).str.contains(r'\d', na=False).mean() > 0.5:
                df[col] = df[col].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)').astype(float)
        return df
    return pd.DataFrame()

df_raw = load_default_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/ef/University_of_Buea_logo.png/220px-University_of_Buea_logo.png", use_container_width=True)
st.sidebar.markdown("<h2 style='text-align: center; color: #003366;'>QoE Simulator</h2>", unsafe_allow_html=True)

menu = ["🏠 Dashboard Home", "📊 Dataset & EDA", "🔮 Scenario Simulator", "📈 Batch Predictions"]
choice = st.sidebar.radio("Navigation", menu)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; color: #008000;'>Telecom ML Research Project</p>", unsafe_allow_html=True)

# --- MAIN SECTIONS ---

if choice == "🏠 Dashboard Home":
    st.title("📶 Telecom QoE Prediction Dashboard")
    st.markdown("""
    Welcome to the **Quality of Experience (QoE)** prediction dashboard, built for the **University of Buea** Master's thesis project. 
    This tool allows network engineers to simulate how various Quality of Service (QoS) parameters impact the perceived user experience in Cameroon's telecom networks.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records Analyzed", len(df_raw) if not df_raw.empty else 0)
    col2.metric("Prediction Model", "XGBoost Regressor" if best_model else "Not Loaded")
    col3.metric("Dataset Region", "Cameroon")

    if not best_model:
        st.warning("⚠️ Models not found. Please run the Jupyter Notebook first to generate and save `best_model.pkl` and `preprocessor.pkl`.")

elif choice == "📊 Dataset & EDA":
    st.title("📊 Exploratory Data Analysis")
    
    if not df_raw.empty:
        st.subheader("Dataset Overview")
        st.dataframe(df_raw.head(10))
        
        st.subheader("QoE Distribution")
        target_col = [c for c in df_raw.columns if 'qoe' in c.lower()]
        if target_col:
            fig = px.histogram(df_raw, x=target_col[0], nbins=20, title="Distribution of QoE Scores",
                               color_discrete_sequence=['#003366'])
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Correlation Matrix (Numeric KPIs)")
        num_df = df_raw.select_dtypes(include=[np.number])
        if not num_df.empty:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale='Greens', title="Pearson Correlation")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dataset 'Dataset_QoE_Cameroon.csv' not found in the current directory.")

elif choice == "🔮 Scenario Simulator":
    st.title("🔮 Real-Time Telecom Scenario Simulator")
    st.markdown("Adjust the QoS sliders below to simulate network conditions and observe the predicted QoE score instantly.")
    
    if not preprocessor or not best_model:
        st.error("Model files missing. Cannot run simulator.")
    else:
        # Create input form based on feature names
        # Since we don't know the exact names beforehand, we will try to match common ones
        # and provide default sliders for the rest.
        
        # We need a dummy dataframe to pass to the preprocessor
        # We will use median values from df_raw for defaults
        if not df_raw.empty:
            input_data = {}
            col1, col2 = st.columns(2)
            
            # Separate numeric and categorical
            num_cols = df_raw.select_dtypes(include=[np.number]).columns.drop([c for c in df_raw.columns if 'qoe' in c.lower()], errors='ignore')
            cat_cols = df_raw.select_dtypes(include=['object']).columns
            
            with col1:
                st.subheader("Core QoS Parameters")
                for col in num_cols:
                    if any(k in col.lower() for k in ['latency', 'jitter', 'loss', 'dl', 'ul', 'mbps', 'bandwidth']):
                        min_val = float(df_raw[col].min())
                        max_val = float(df_raw[col].max())
                        mean_val = float(df_raw[col].median())
                        input_data[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
                        
            with col2:
                st.subheader("Other & Categorical Parameters")
                for col in num_cols:
                    if col not in input_data:
                        mean_val = float(df_raw[col].median())
                        input_data[col] = mean_val # Hidden/default for others to simplify UI
                
                for col in cat_cols:
                    options = df_raw[col].dropna().unique()
                    if len(options) > 0:
                        input_data[col] = st.selectbox(f"{col}", options=options)
                    else:
                        input_data[col] = "Unknown"
                        
            # Engineer features if we had them in notebook
            dl_col = next((c for c in num_cols if 'dl' in c.lower() or 'download' in c.lower()), None)
            ul_col = next((c for c in num_cols if 'ul' in c.lower() or 'upload' in c.lower()), None)
            jitter_col = next((c for c in num_cols if 'jitter' in c.lower()), None)
            loss_col = next((c for c in num_cols if 'loss' in c.lower()), None)
            
            if dl_col and ul_col:
                input_data['Throughput_Ratio'] = input_data[dl_col] / (input_data[ul_col] + 1e-5)
            if jitter_col and loss_col:
                # Approximate normalization
                j_mean = df_raw[jitter_col].mean()
                j_std = df_raw[jitter_col].std() or 1
                l_mean = df_raw[loss_col].mean()
                l_std = df_raw[loss_col].std() or 1
                input_data['Network_Instability_Index'] = ((input_data[jitter_col] - j_mean)/j_std) + ((input_data[loss_col] - l_mean)/l_std)

            # Create dataframe
            input_df = pd.DataFrame([input_data])
            
            # Predict
            try:
                processed_input = preprocessor.transform(input_df)
                prediction = best_model.predict(processed_input)[0]
                
                st.markdown("---")
                st.markdown("<h3 style='text-align: center;'>Predicted QoE Score</h3>", unsafe_allow_html=True)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "QoE (MOS)"},
                    gauge = {'axis': {'range': [1, 5]},
                             'bar': {'color': "#003366"},
                             'steps' : [
                                 {'range': [1, 2.5], 'color': "red"},
                                 {'range': [2.5, 3.5], 'color': "yellow"},
                                 {'range': [3.5, 5], 'color': "#008000"}],
                            }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # SHAP Explainability
                st.subheader("🧠 Why this prediction? (SHAP Explainability)")
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(processed_input)
                
                shap.force_plot(explainer.expected_value, shap_values[0,:], processed_input[0,:], feature_names=feature_names, matplotlib=True)
                st.pyplot(plt.gcf(), clear_figure=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}. Please ensure all features match the training data.")
        else:
            st.warning("Need Dataset_QoE_Cameroon.csv to build the simulator UI based on column names.")

elif choice == "📈 Batch Predictions":
    st.title("📈 Batch Predictions (Upload New Data)")
    st.markdown("Upload a CSV file with new telecom KPI logs to predict QoE scores for all rows.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        if preprocessor and best_model:
            batch_df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='utf-8', engine='python')
            batch_df.columns = batch_df.columns.str.strip()
            for col in batch_df.select_dtypes(include=['object']).columns:
                if batch_df[col].astype(str).str.contains(r'\d', na=False).mean() > 0.5:
                    batch_df[col] = batch_df[col].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)').astype(float)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())
            
            if st.button("Predict QoE"):
                try:
                    # Apply same engineering
                    # Assuming basic columns match
                    dl_col = next((c for c in batch_df.columns if 'dl' in c.lower() or 'download' in c.lower()), None)
                    ul_col = next((c for c in batch_df.columns if 'ul' in c.lower() or 'upload' in c.lower()), None)
                    jitter_col = next((c for c in batch_df.columns if 'jitter' in c.lower()), None)
                    loss_col = next((c for c in batch_df.columns if 'loss' in c.lower()), None)
                    
                    if dl_col and ul_col:
                        batch_df['Throughput_Ratio'] = batch_df[dl_col] / (batch_df[ul_col] + 1e-5)
                    if jitter_col and loss_col:
                        j_mean = df_raw[jitter_col].mean() if not df_raw.empty else batch_df[jitter_col].mean()
                        j_std = df_raw[jitter_col].std() if not df_raw.empty else batch_df[jitter_col].std()
                        l_mean = df_raw[loss_col].mean() if not df_raw.empty else batch_df[loss_col].mean()
                        l_std = df_raw[loss_col].std() if not df_raw.empty else batch_df[loss_col].std()
                        batch_df['Network_Instability_Index'] = ((batch_df[jitter_col] - j_mean)/j_std) + ((batch_df[loss_col] - l_mean)/l_std)

                    X_batch = preprocessor.transform(batch_df)
                    preds = best_model.predict(X_batch)
                    batch_df['Predicted_QoE'] = preds
                    
                    st.success("Predictions complete!")
                    st.dataframe(batch_df)
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='QoE_Predictions_Output.csv',
                        mime='text/csv',
                    )
                except Exception as e:
                    st.error(f"Error during prediction pipeline: {e}")
        else:
            st.error("Model files are not available.")
