"""
app.py - Streamlit Dashboard for Waterborne Disease Outbreak Prediction
========================================================================
A production-grade, interactive AI dashboard that predicts waterborne
disease outbreaks across India using a trained XGBoost model.

Run: streamlit run app/app.py
"""

# import os
# import sys
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.express as px

# # ──────────────────────────────────────────────────────────────
# # PATH SETUP
# # ──────────────────────────────────────────────────────────────
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_DIR)

# MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
# ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")
# RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "waterborne_disease.csv")
# PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "waterborne_processed.csv")

# from app.components import (
#     create_gauge_chart,
#     create_feature_importance_chart,
#     create_confusion_matrix_chart,
#     create_outbreak_distribution_chart,
#     create_prediction_result_card
# )

# ──────────────────────────────────────────────────────────────
# PATH SETUP (Robust for Cloud & Local)
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure the root directory is in sys.path so 'app' and 'models' are findable
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Dynamically resolve paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "waterborne_disease.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "waterborne_processed.csv")

# IMPORT COMPONENTS (Handling both local and cloud pathing)
try:
    from components import (
        create_gauge_chart, create_feature_importance_chart,
        create_confusion_matrix_chart, create_outbreak_distribution_chart,
        create_prediction_result_card
    )
except ImportError:
    from app.components import (
        create_gauge_chart, create_feature_importance_chart,
        create_confusion_matrix_chart, create_outbreak_distribution_chart,
        create_prediction_result_card
    )

    

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaShield AI – Waterborne Disease Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid rgba(0, 204, 102, 0.2);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .main-header h1 {
        background: linear-gradient(90deg, #00b359, #0088cc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #555555;
        font-size: 1.1rem;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 204, 102, 0.4);
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00b359;
    }
    .metric-label {
        color: #555555;
        font-size: 0.9rem;
        margin-top: 5px;
    }

    /* Section headers */
    .section-header {
        color: #0088cc;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0, 136, 204, 0.2);
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# DATA & MODEL LOADING (Cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at `{MODEL_PATH}`. Please run `python src/model_trainer.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_encoders():
    """Load the label encoders for categorical features."""
    if not os.path.exists(ENCODERS_PATH):
        return None
    return joblib.load(ENCODERS_PATH)


@st.cache_data
def load_data():
    """Load the raw dataset for exploratory analysis."""
    if os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH, nrows=100000)  # Load subset for speed
        df['Outbreak_Risk'] = np.where(df['disease'] == 'No_Disease', 0, 1)
        return df
    return None


# ──────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #00b359; margin: 0;">🛡️ AquaShield AI</h2>
        <p style="color: #555555; font-size: 0.85rem;">Waterborne Disease Predictor</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🔮 Predict Outbreak", "📊 Data Explorer", "📈 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; color: #484f58; font-size: 0.75rem; padding: 10px;">
        <p>Built with ❤️ using XGBoost</p>
        <p>Dataset: 5.25M Records</p>
        <p>Coverage: All India</p>
    </div>
    """, unsafe_allow_html=True)


# Load resources
model = load_model()
encoders = load_encoders()
df = load_data()


# ──────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ──────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ AquaShield AI</h1>
        <p>AI-Powered Waterborne Disease Outbreak Prediction System for India</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5.25M</div>
            <div class="metric-label">📋 Training Records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">88.44%</div>
            <div class="metric-label">🎯 Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">9</div>
            <div class="metric-label">📊 Input Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">XGBoost</div>
            <div class="metric-label">🤖 AI Engine</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    if df is not None:
        col_left, col_right = st.columns(2)

        with col_left:
            fig = create_outbreak_distribution_chart(df['Outbreak_Risk'])
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Feature importance from model
            feature_names = [
                'is_urban', 'population_density', 'water_source', 'water_treatment',
                'ph', 'avg_temperature_c', 'avg_rainfall_mm', 'avg_humidity_pct', 'flooding'
            ]
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig = create_feature_importance_chart(fi_df)
            st.plotly_chart(fig, use_container_width=True)

    # Project overview
    st.markdown('<div class="section-header">📋 About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    > **AquaShield AI** is a machine learning system trained on **5.25 million records**
    > covering all states and districts of India. It predicts the risk of waterborne disease
    > outbreaks based on water quality, sanitation, and environmental factors.

    **Key Features:**
    - 🔮 **Real-time Prediction** – Enter area parameters to get instant outbreak risk assessment
    - 📊 **Data Explorer** – Analyze patterns across India's water quality data
    - 🤖 **XGBoost Engine** – Histogram-based gradient boosting optimized for big data
    - 📈 **Model Insights** – Feature importance and performance metrics
    """)


# ──────────────────────────────────────────────────────────────
# PAGE: PREDICT OUTBREAK
# ──────────────────────────────────────────────────────────────
elif page == "🔮 Predict Outbreak":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Predict Outbreak Risk</h1>
        <p>Enter the environmental and water quality parameters for any region</p>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    with st.form("prediction_form"):
        st.markdown('<div class="section-header">📝 Area Parameters</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            is_urban = st.selectbox("🏙️ Area Type", [("Urban", 1), ("Rural", 0)], format_func=lambda x: x[0])
            population_density = st.number_input("👥 Population Density (per km²)", min_value=0, max_value=50000, value=500, step=50)
            water_source = st.selectbox("💧 Water Source", ["Piped", "Borewell", "River", "Pond", "Tanker"])

        with col2:
            water_treatment = st.selectbox("🧪 Water Treatment", ["Chlorination", "Filtration", "Boiling", "No_Treatment", "UV_Treatment"])
            ph = st.slider("⚗️ pH Level", min_value=4.0, max_value=10.0, value=7.0, step=0.1)
            avg_temperature_c = st.slider("🌡️ Avg Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0, step=0.5)

        with col3:
            avg_rainfall_mm = st.number_input("🌧️ Avg Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
            avg_humidity_pct = st.slider("💨 Avg Humidity (%)", min_value=20.0, max_value=100.0, value=70.0, step=1.0)
            flooding = st.selectbox("🌊 Flooding", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])

        submitted = st.form_submit_button("🔮 Predict Outbreak Risk", use_container_width=True)

    if submitted:
        # Encode categorical features
        if encoders:
            try:
                ws_encoded = encoders['water_source'].transform([water_source])[0]
                wt_encoded = encoders['water_treatment'].transform([water_treatment])[0]
            except ValueError:
                ws_encoded = 0
                wt_encoded = 0
        else:
            # Fallback manual encoding
            ws_map = {"Piped": 0, "Borewell": 1, "River": 2, "Pond": 3, "Tanker": 4}
            wt_map = {"Chlorination": 0, "Filtration": 1, "Boiling": 2, "No_Treatment": 3, "UV_Treatment": 4}
            ws_encoded = ws_map.get(water_source, 0)
            wt_encoded = wt_map.get(water_treatment, 0)

        # Build input dataframe
        input_data = pd.DataFrame({
            'is_urban': [is_urban[1]],
            'population_density': [population_density],
            'water_source': [ws_encoded],
            'water_treatment': [wt_encoded],
            'ph': [ph],
            'avg_temperature_c': [avg_temperature_c],
            'avg_rainfall_mm': [avg_rainfall_mm],
            'avg_humidity_pct': [avg_humidity_pct],
            'flooding': [flooding[1]]
        })

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Display result
        st.markdown(create_prediction_result_card(prediction, probability), unsafe_allow_html=True)

        # Show input summary
        with st.expander("📋 View Input Parameters"):
            st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ──────────────────────────────────────────────────────────────
elif page == "📊 Data Explorer":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Explorer</h1>
        <p>Explore patterns in India's waterborne disease data (Showing 100K sample)</p>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records (Sample)", f"{len(df):,}")
        with col2:
            st.metric("States Covered", df['state'].nunique() if 'state' in df.columns else "N/A")
        with col3:
            outbreak_pct = (df['Outbreak_Risk'].sum() / len(df)) * 100
            st.metric("Outbreak Rate", f"{outbreak_pct:.1f}%")

        st.markdown("---")

        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Distribution Analysis", "🗺️ Geographic Insights", "🔬 Correlation"])

        with tab1:
            col_left, col_right = st.columns(2)

            with col_left:
                feature = st.selectbox("Select Feature", [
                    'ph', 'population_density', 'avg_temperature_c',
                    'avg_rainfall_mm', 'avg_humidity_pct', 'water_source', 'water_treatment'
                ])

                if feature in ['water_source', 'water_treatment']:
                    fig = px.histogram(df, x=feature, color='Outbreak_Risk',
                                       barmode='group',
                                       color_discrete_map={0: '#00b359', 1: '#ff4444'},
                                       template='plotly_white')
                else:
                    fig = px.histogram(df, x=feature, color='Outbreak_Risk',
                                       marginal='box',
                                       color_discrete_map={0: '#00b359', 1: '#ff4444'},
                                       template='plotly_white')

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                fig = create_outbreak_distribution_chart(df['Outbreak_Risk'])
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if 'state' in df.columns:
                state_data = df.groupby('state')['Outbreak_Risk'].mean().reset_index()
                state_data.columns = ['State', 'Outbreak_Rate']
                state_data = state_data.sort_values('Outbreak_Rate', ascending=False)

                fig = px.bar(state_data.head(20), x='Outbreak_Rate', y='State',
                             orientation='h',
                             color='Outbreak_Rate',
                             color_continuous_scale='RdYlGn_r',
                             template='plotly_white',
                             title='Top 20 States by Outbreak Risk Rate')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            numeric_cols = df[['ph', 'population_density', 'avg_temperature_c',
                               'avg_rainfall_mm', 'avg_humidity_pct', 'Outbreak_Risk']].corr()
            fig = px.imshow(numeric_cols.values,
                            x=numeric_cols.columns,
                            y=numeric_cols.index,
                            text_auto=".2f",
                            color_continuous_scale='RdBu_r',
                            template='plotly_white',
                            title='Feature Correlation Heatmap')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        # Raw data preview
        st.markdown('<div class="section-header">📄 Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True, height=300)
    else:
        st.warning("⚠️ Raw dataset not found. Please ensure the CSV is at `data/raw/waterborne_disease.csv`.")


# ──────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Model Performance</h1>
        <p>XGBoost classifier metrics and insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Accuracy gauge
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        fig = create_gauge_chart(88.44, "Overall Accuracy")
        st.plotly_chart(fig, use_container_width=True)

    # Classification report
    st.markdown('<div class="section-header">📋 Classification Report</div>', unsafe_allow_html=True)

    report_data = {
        'Class': ['Safe (0)', 'Outbreak (1)', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.86, 0.90, 0.88, 0.88],
        'Recall': [0.86, 0.90, 0.88, 0.88],
        'F1-Score': [0.86, 0.90, 0.88, 0.88],
        'Support': ['419,666', '630,334', '1,050,000', '1,050,000']
    }
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # Feature importance
    st.markdown('<div class="section-header">🏆 Feature Importance</div>', unsafe_allow_html=True)

    feature_names = [
        'is_urban', 'population_density', 'water_source', 'water_treatment',
        'ph', 'avg_temperature_c', 'avg_rainfall_mm', 'avg_humidity_pct', 'flooding'
    ]
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig = create_feature_importance_chart(fi_df)
    st.plotly_chart(fig, use_container_width=True)

    # Model config
    st.markdown('<div class="section-header">⚙️ Model Configuration</div>', unsafe_allow_html=True)
    config_data = {
        'Parameter': ['Algorithm', 'n_estimators', 'learning_rate', 'max_depth', 'tree_method', 'eval_metric'],
        'Value': ['XGBClassifier', '100', '0.1', '6', 'hist', 'logloss']
    }
    st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)
