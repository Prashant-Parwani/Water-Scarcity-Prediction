"""
================================================================================
WATER SCARCITY PREDICTION SYSTEM - STREAMLIT APP (FULLY FIXED VERSION)
Now Actually Uses Trained ML Models for Forecasting
================================================================================

INSTALLATION:
pip install streamlit pandas numpy plotly scikit-learn xgboost

RUN:
streamlit run water_scarcity_app_fixed.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Water Scarcity Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1.5rem;
        border-radius: 8px;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1.5rem;
        border-radius: 8px;
    }
    .alert-normal {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1.5rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    try:
        yearly_path = r"F:\Water Scarcity Prediction\DATA FILES\Clean Datasets\FINAL_YEARLY_WATER_SCARCITY_DATASET.csv"
        monthly_path = r"F:\Water Scarcity Prediction\DATA FILES\Clean Datasets\FINAL_MONTHLY_WATER_SCARCITY_DATASET.csv"
        
        df_yearly = pd.read_csv(yearly_path)
        df_monthly = pd.read_csv(monthly_path)
        
        return df_yearly, df_monthly
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    models = {}
    model_path = r"F:\Water Scarcity Prediction\DATA FILES\ML Models"
    
    try:
        # Load XGBoost Classification Model
        with open(os.path.join(model_path, 'xgboost_classifier_final.pkl'), 'rb') as f:
            models['xgboost'] = pickle.load(f)
        with open(os.path.join(model_path, 'xgb_label_encoder.pkl'), 'rb') as f:
            models['label_encoder'] = pickle.load(f)
        with open(os.path.join(model_path, 'xgb_scaler.pkl'), 'rb') as f:
            models['xgb_scaler'] = pickle.load(f)
        with open(os.path.join(model_path, 'xgb_features.pkl'), 'rb') as f:
            models['xgb_features'] = pickle.load(f)
        st.sidebar.success("✓ XGBoost model loaded")
    except Exception as e:
        st.sidebar.warning(f"XGBoost not loaded: {e}")
    
    try:
        # Load Ensemble Regression Models
        with open(os.path.join(model_path, 'gradient_boosting_final.pkl'), 'rb') as f:
            models['gradient_boosting'] = pickle.load(f)
        with open(os.path.join(model_path, 'random_forest_final.pkl'), 'rb') as f:
            models['random_forest'] = pickle.load(f)
        with open(os.path.join(model_path, 'extra_trees_final.pkl'), 'rb') as f:
            models['extra_trees'] = pickle.load(f)
        with open(os.path.join(model_path, 'ensemble_scaler_X.pkl'), 'rb') as f:
            models['ensemble_scaler_X'] = pickle.load(f)
        with open(os.path.join(model_path, 'ensemble_scaler_y.pkl'), 'rb') as f:
            models['ensemble_scaler_y'] = pickle.load(f)
        with open(os.path.join(model_path, 'ensemble_features.pkl'), 'rb') as f:
            models['ensemble_features'] = pickle.load(f)
        st.sidebar.success("✓ Ensemble models loaded")
    except Exception as e:
        st.sidebar.warning(f"Ensemble not loaded: {e}")
    
    return models

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_scarcity_info(index):
    """Get scarcity level information"""
    if index < 0.5:
        return "Low", "🟢", "#4caf50", "alert-normal"
    elif index < 1.0:
        return "Medium", "🟡", "#ff9800", "alert-warning"
    elif index < 1.5:
        return "High", "🟠", "#ff5722", "alert-warning"
    else:
        return "Critical", "🔴", "#f44336", "alert-critical"

def get_recommendations(level):
    """Get recommendations based on scarcity level"""
    recommendations = {
        "Low": [
            "✓ Continue water conservation practices",
            "✓ Monitor groundwater levels regularly",
            "✓ Maintain rainwater harvesting systems",
            "✓ Encourage sustainable agriculture"
        ],
        "Medium": [
            "⚠ Prepare water conservation plans",
            "⚠ Increase public awareness campaigns",
            "⚠ Implement water-efficient agriculture",
            "⚠ Monitor extraction rates closely"
        ],
        "High": [
            "⚠️ Implement water rationing measures",
            "⚠️ Accelerate groundwater recharge projects",
            "⚠️ Restrict non-essential water use",
            "⚠️ Deploy water-saving technologies"
        ],
        "Critical": [
            "🚨 IMMEDIATE ACTION REQUIRED",
            "🚨 Deploy emergency water tankers",
            "🚨 Enforce strict water rationing",
            "🚨 Activate emergency response protocols",
            "🚨 Consider alternative water sources"
        ]
    }
    return recommendations.get(level, [])

# ============================================================================
# FIXED FORECASTING FUNCTIONS - NOW ACTUALLY USE TRAINED MODELS
# ============================================================================

def prepare_ensemble_features_for_forecast(district_monthly_data, forecast_year, forecast_month=6):
    """
    Prepare features matching the training ensemble features
    This recreates all the lag, rolling, and interaction features
    """
    # Get the most recent 24 months of data (need for 12-month lags)
    df = district_monthly_data.copy()
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    df = df.sort_values('Date').tail(24)
    
    if len(df) == 0:
        return None
    
    # Create a new row for forecasting
    last_row = df.iloc[-1].copy()
    
    # Basic features (use latest values or projections)
    features = {}
    
    # Base features
    features['Rainfall_mm'] = df['Rainfall_mm'].tail(12).mean()  # Use 12-month avg
    features['GWL_meters'] = last_row['GWL_meters']
    features['GWL_Index'] = last_row['GWL_Index']
    features['Water_Demand_MCM'] = last_row['Water_Demand_MCM'] * 1.02  # 2% growth
    features['Water_Supply_MCM'] = last_row['Water_Supply_MCM'] * 0.99  # Slight decline
    
    # Lag features (from historical data)
    for lag in [1, 2, 3, 6, 12]:
        if len(df) >= lag:
            features[f'Scarcity_Lag_{lag}'] = df['Scarcity_Index'].iloc[-lag]
            features[f'Rainfall_Lag_{lag}'] = df['Rainfall_mm'].iloc[-lag] if lag <= len(df) else df['Rainfall_mm'].mean()
            features[f'GWL_Lag_{lag}'] = df['GWL_meters'].iloc[-lag] if lag <= len(df) else df['GWL_meters'].mean()
            features[f'Supply_Lag_{lag}'] = df['Water_Supply_MCM'].iloc[-lag] if lag <= len(df) else df['Water_Supply_MCM'].mean()
            features[f'Demand_Lag_{lag}'] = df['Water_Demand_MCM'].iloc[-lag] if lag <= len(df) else df['Water_Demand_MCM'].mean()
        else:
            # Not enough history, use latest value
            features[f'Scarcity_Lag_{lag}'] = df['Scarcity_Index'].iloc[-1]
            features[f'Rainfall_Lag_{lag}'] = df['Rainfall_mm'].iloc[-1]
            features[f'GWL_Lag_{lag}'] = df['GWL_meters'].iloc[-1]
            features[f'Supply_Lag_{lag}'] = df['Water_Supply_MCM'].iloc[-1]
            features[f'Demand_Lag_{lag}'] = df['Water_Demand_MCM'].iloc[-1]
    
    # Rolling statistics (use recent history)
    for window in [3, 6, 12]:
        window_data = df.tail(min(window, len(df)))
        features[f'Scarcity_MA_{window}'] = window_data['Scarcity_Index'].mean()
        features[f'Scarcity_STD_{window}'] = window_data['Scarcity_Index'].std() if len(window_data) > 1 else 0
        features[f'Rainfall_MA_{window}'] = window_data['Rainfall_mm'].mean()
        features[f'Rainfall_STD_{window}'] = window_data['Rainfall_mm'].std() if len(window_data) > 1 else 0
        features[f'GWL_MA_{window}'] = window_data['GWL_meters'].mean()
        features[f'Supply_MA_{window}'] = window_data['Water_Supply_MCM'].mean()
    
    # Trends (differences)
    features['Scarcity_Trend_3'] = df['Scarcity_Index'].diff(3).iloc[-1] if len(df) >= 4 else 0
    features['Scarcity_Trend_6'] = df['Scarcity_Index'].diff(6).iloc[-1] if len(df) >= 7 else 0
    features['Scarcity_Trend_12'] = df['Scarcity_Index'].diff(12).iloc[-1] if len(df) >= 13 else 0
    features['Rainfall_Trend_3'] = df['Rainfall_mm'].diff(3).iloc[-1] if len(df) >= 4 else 0
    features['Rainfall_Trend_6'] = df['Rainfall_mm'].diff(6).iloc[-1] if len(df) >= 7 else 0
    
    # Seasonal features
    features['Month_Sin'] = np.sin(2 * np.pi * forecast_month / 12)
    features['Month_Cos'] = np.cos(2 * np.pi * forecast_month / 12)
    features['Is_Monsoon'] = 1 if forecast_month in [6, 7, 8, 9] else 0
    features['Is_Summer'] = 1 if forecast_month in [3, 4, 5] else 0
    features['Is_Winter'] = 1 if forecast_month in [11, 12, 1, 2] else 0
    features['Is_PostMonsoon'] = 1 if forecast_month in [10, 11] else 0
    
    # Year features
    year_min = df['Year'].min()
    year_max = df['Year'].max()
    year_range = year_max - year_min if year_max > year_min else 1
    features['Year_Normalized'] = (forecast_year - year_min) / year_range
    features['Year_Squared'] = features['Year_Normalized'] ** 2
    
    # Interaction features
    features['Demand_Supply_Ratio'] = features['Water_Demand_MCM'] / (features['Water_Supply_MCM'] + 0.001)
    features['Rainfall_GWL_Interaction'] = features['Rainfall_mm'] * features['GWL_Index']
    features['Monsoon_Rainfall'] = features['Is_Monsoon'] * features['Rainfall_mm']
    features['Supply_Demand_Gap'] = features['Water_Supply_MCM'] - features['Water_Demand_MCM']
    
    # YoY changes
    if len(df) >= 13:
        features['Scarcity_YoY'] = df['Scarcity_Index'].iloc[-1] - df['Scarcity_Index'].iloc[-13]
        features['Rainfall_YoY'] = df['Rainfall_mm'].iloc[-1] - df['Rainfall_mm'].iloc[-13]
    else:
        features['Scarcity_YoY'] = 0
        features['Rainfall_YoY'] = 0
    
    return features


def forecast_with_actual_ensemble(models, district_data_yearly, district_data_monthly, years_ahead=15):
    """
    FIXED: Actually use the trained ensemble models for forecasting
    """
    if not all(k in models for k in ['gradient_boosting', 'random_forest', 'extra_trees', 
                                      'ensemble_scaler_X', 'ensemble_scaler_y', 'ensemble_features']):
        return None
    
    if len(district_data_monthly) < 12:
        return None
    
    forecasts = []
    current_year = district_data_yearly['Year'].max()
    
    # Make a working copy of monthly data
    working_monthly = district_data_monthly.copy()
    
    # For each forecast year
    for i in range(1, years_ahead + 1):
        forecast_year = current_year + i
        
        # Forecast for mid-year (June) as representative
        forecast_month = 6
        
        # Prepare features
        feature_dict = prepare_ensemble_features_for_forecast(
            working_monthly, forecast_year, forecast_month
        )
        
        if feature_dict is None:
            continue
        
        # Arrange features in correct order
        feature_values = []
        for feat_name in models['ensemble_features']:
            feature_values.append(feature_dict.get(feat_name, 0))
        
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        X_scaled = models['ensemble_scaler_X'].transform(X)
        
        # Predict with each model
        gb_pred = models['gradient_boosting'].predict(X_scaled)
        rf_pred = models['random_forest'].predict(X_scaled)
        et_pred = models['extra_trees'].predict(X_scaled)
        
        # Weighted ensemble (same weights as training)
        y_pred_scaled = 0.45 * gb_pred + 0.35 * rf_pred + 0.20 * et_pred
        
        # Inverse transform
        y_pred = models['ensemble_scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
        
        # Clip to valid range
        y_pred = np.clip(y_pred, 0.1, 10.0)
        
        # Determine level
        level, _, _, _ = get_scarcity_info(y_pred)
        
        # Confidence decreases with time
        confidence = max(50, 95 - (i * 2.5))
        
        forecasts.append({
            'Year': forecast_year,
            'Scarcity_Index': round(y_pred, 3),
            'Scarcity_Level': level,
            'Model': 'Ensemble',
            'Confidence': round(confidence, 1),
            'GB_Prediction': round(gb_pred[0], 3),
            'RF_Prediction': round(rf_pred[0], 3),
            'ET_Prediction': round(et_pred[0], 3)
        })
        
        # Update working_monthly with this prediction for next iteration
        new_row = working_monthly.iloc[-1].copy()
        new_row['Year'] = forecast_year
        new_row['Month'] = forecast_month
        new_row['Scarcity_Index'] = y_pred
        new_row['Water_Demand_MCM'] = new_row['Water_Demand_MCM'] * 1.02
        new_row['Water_Supply_MCM'] = new_row['Water_Supply_MCM'] * 0.99
        working_monthly = pd.concat([working_monthly, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(forecasts)


def prepare_xgb_features_for_forecast(yearly_data, forecast_year, xgb_features_list):
    """
    Prepare features matching the training XGBoost features
    """
    last_row = yearly_data.iloc[-1]
    
    features = {}
    
    # Base features - use latest or project forward
    years_ahead = forecast_year - last_row['Year']
    
    if 'Normal_Annual_Rainfall' in yearly_data.columns:
        features['Normal_Annual_Rainfall'] = last_row['Normal_Annual_Rainfall']
    if 'Annual_Rainfall' in yearly_data.columns:
        features['Annual_Rainfall'] = yearly_data['Annual_Rainfall'].tail(5).mean()
    if 'Rainfall_Anomaly' in yearly_data.columns:
        features['Rainfall_Anomaly'] = yearly_data['Rainfall_Anomaly'].tail(5).mean()
    
    if 'Net_GW_Available_MCM' in yearly_data.columns:
        features['Net_GW_Available_MCM'] = last_row['Net_GW_Available_MCM'] * (0.98 ** years_ahead)
    if 'Total_Annual_Draft_MCM' in yearly_data.columns:
        features['Total_Annual_Draft_MCM'] = last_row['Total_Annual_Draft_MCM'] * (1.02 ** years_ahead)
    if 'Stage_of_Extraction' in yearly_data.columns:
        features['Stage_of_Extraction'] = last_row['Stage_of_Extraction'] * (1.01 ** years_ahead)
    
    if 'Recharge_Monsoon' in yearly_data.columns:
        features['Recharge_Monsoon'] = last_row['Recharge_Monsoon']
    if 'Recharge_NonMonsoon' in yearly_data.columns:
        features['Recharge_NonMonsoon'] = last_row['Recharge_NonMonsoon']
    
    # Project demand/supply
    if 'Water_Demand_MCM' in yearly_data.columns:
        features['Water_Demand_MCM'] = last_row['Water_Demand_MCM'] * (1.02 ** years_ahead)
    if 'Water_Supply_MCM' in yearly_data.columns:
        features['Water_Supply_MCM'] = last_row['Water_Supply_MCM'] * (0.99 ** years_ahead)
    if 'Water_Deficit_MCM' in yearly_data.columns:
        features['Water_Deficit_MCM'] = features.get('Water_Demand_MCM', 0) - features.get('Water_Supply_MCM', 0)
    
    if 'GWL_meters' in yearly_data.columns:
        features['GWL_meters'] = last_row['GWL_meters'] + (0.1 * years_ahead)
    if 'GWL_Index' in yearly_data.columns:
        features['GWL_Index'] = last_row['GWL_Index'] * (1.01 ** years_ahead)
    if 'GWL_Trend' in yearly_data.columns:
        features['GWL_Trend'] = yearly_data['GWL_Trend'].tail(3).mean() if len(yearly_data) >= 3 else 0
    
    # Population projection
    if 'Total_Population' in yearly_data.columns:
        features['Total_Population'] = last_row['Total_Population'] * (1.012 ** years_ahead)
    if 'Adjusted_Population' in yearly_data.columns:
        features['Adjusted_Population'] = last_row['Adjusted_Population'] * (1.012 ** years_ahead)
    
    # Derived features (matching training code)
    monsoon = features.get('Recharge_Monsoon', 0)
    non_monsoon = features.get('Recharge_NonMonsoon', 0)
    features['Monsoon_Dependency'] = monsoon / (monsoon + non_monsoon + 0.001)
    
    supply = features.get('Water_Supply_MCM', 1)
    demand = features.get('Water_Demand_MCM', 1)
    features['Supply_Demand_Ratio'] = supply / (demand + 0.001)
    features['Deficit_Percentage'] = (features.get('Water_Deficit_MCM', 0) / (demand + 0.001)) * 100
    
    features['Extraction_Pressure'] = features.get('Total_Annual_Draft_MCM', 0) / (features.get('Net_GW_Available_MCM', 1) + 0.001)
    features['Water_Stress_Index'] = features.get('Stage_of_Extraction', 0) * abs(features.get('Deficit_Percentage', 0)) / 100
    features['Per_Capita_Water'] = supply / (features.get('Total_Population', 1) + 1) * 1000000
    features['GWL_Stress'] = features.get('GWL_Index', 0) * features.get('Stage_of_Extraction', 0)
    
    # Year normalization
    year_min = yearly_data['Year'].min()
    year_max = yearly_data['Year'].max()
    features['Year_Normalized'] = (forecast_year - year_min) / (year_max - year_min) if year_max > year_min else 0
    
    # Rainfall features if available
    if 'Rainfall_Anomaly' in features:
        features['Rainfall_Stress'] = 1 / (features['Rainfall_Anomaly'] + 0.001)
    if 'Normal_Annual_Rainfall' in features and 'Annual_Rainfall' in features:
        features['Rainfall_Deficit'] = features['Normal_Annual_Rainfall'] - features['Annual_Rainfall']
    
    return features


def forecast_with_actual_xgboost(models, district_data, years_ahead=15):
    """
    FIXED: Actually use the trained XGBoost model for forecasting
    """
    if 'xgboost' not in models or 'xgb_scaler' not in models or 'xgb_features' not in models:
        return None
    
    forecasts = []
    current_year = district_data['Year'].max()
    
    # Working copy for iterative forecasting
    working_data = district_data.copy()
    
    for i in range(1, years_ahead + 1):
        forecast_year = current_year + i
        
        # Prepare features
        feature_dict = prepare_xgb_features_for_forecast(working_data, forecast_year, models['xgb_features'])
        
        # Arrange in correct order
        feature_values = []
        for feat_name in models['xgb_features']:
            feature_values.append(feature_dict.get(feat_name, 0))
        
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale
        X_scaled = models['xgb_scaler'].transform(X)
        
        # Predict class probabilities
        y_pred_proba = models['xgboost'].predict_proba(X_scaled)[0]
        y_pred_class = models['xgboost'].predict(X_scaled)[0]
        
        # Get predicted level name
        level_name = models['label_encoder'].inverse_transform([y_pred_class])[0]
        
        # Map level to scarcity index (approximate midpoint of range)
        level_to_index = {
            'Low': 0.3,
            'Medium': 0.75,
            'High': 1.25,
            'Critical': 2.0
        }
        predicted_index = level_to_index.get(level_name, 1.0)
        
        # Confidence
        confidence = max(50, 92 - (i * 2.3))
        max_prob = y_pred_proba.max()
        
        forecasts.append({
            'Year': forecast_year,
            'Scarcity_Index': round(predicted_index, 3),
            'Scarcity_Level': level_name,
            'Model': 'XGBoost',
            'Confidence': round(confidence, 1),
            'Class_Probability': round(max_prob * 100, 1)
        })
        
        # Update working data for next iteration
        new_row = working_data.iloc[-1].copy()
        new_row['Year'] = forecast_year
        new_row['Scarcity_Index'] = predicted_index
        if 'Water_Demand_MCM' in new_row:
            new_row['Water_Demand_MCM'] = new_row['Water_Demand_MCM'] * 1.02
        if 'Water_Supply_MCM' in new_row:
            new_row['Water_Supply_MCM'] = new_row['Water_Supply_MCM'] * 0.99
        working_data = pd.concat([working_data, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(forecasts)


def hybrid_forecast_fixed(models, district_data_yearly, district_data_monthly, years_ahead=15):
    """
    FIXED: Hybrid forecast using actual trained models
    """
    ensemble_fc = forecast_with_actual_ensemble(models, district_data_yearly, district_data_monthly, years_ahead)
    xgb_fc = forecast_with_actual_xgboost(models, district_data_yearly, years_ahead)
    
    if ensemble_fc is None and xgb_fc is None:
        return None
    if ensemble_fc is None:
        return xgb_fc
    if xgb_fc is None:
        return ensemble_fc
    
    # Combine predictions
    combined = []
    for i in range(len(ensemble_fc)):
        ens_val = ensemble_fc.iloc[i]['Scarcity_Index']
        xgb_val = xgb_fc.iloc[i]['Scarcity_Index']
        
        # Weighted average (ensemble better for short-term)
        ensemble_weight = max(0.4, 0.7 - (i * 0.02))
        xgb_weight = 1 - ensemble_weight
        
        combined_val = ens_val * ensemble_weight + xgb_val * xgb_weight
        level, _, _, _ = get_scarcity_info(combined_val)
        
        combined.append({
            'Year': ensemble_fc.iloc[i]['Year'],
            'Scarcity_Index': round(combined_val, 3),
            'Scarcity_Level': level,
            'Ensemble_Pred': round(ens_val, 3),
            'XGBoost_Pred': round(xgb_val, 3),
            'Model': 'Hybrid',
            'Confidence': round((ensemble_fc.iloc[i]['Confidence'] + xgb_fc.iloc[i]['Confidence']) / 2, 1),
            'Ensemble_Weight': round(ensemble_weight * 100, 1),
            'XGBoost_Weight': round(xgb_weight * 100, 1)
        })
    
    return pd.DataFrame(combined)

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

df_yearly, df_monthly = load_data()
models = load_models()

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-header">💧 Water Scarcity Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">AI-Powered Forecasting (2025-2040) | Now Using Actual Trained ML Models</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("🎛️ Control Panel")

if df_yearly is not None:
    
    view_mode = st.sidebar.radio(
        "📊 Select View",
        ["Current Status", "Historical Trends", "🔮 AI Forecast (ML-Powered)", "National Overview"],
        index=2
    )
    
    st.sidebar.markdown("---")
    
    # District selection
    states = sorted(df_yearly['State'].unique())
    selected_state = st.sidebar.selectbox("📍 State", ["All States"] + list(states))
    
    if selected_state == "All States":
        districts = sorted(df_yearly['District'].unique())
    else:
        districts = sorted(df_yearly[df_yearly['State'] == selected_state]['District'].unique())
    
    selected_district = st.sidebar.selectbox("🏘️ District", districts)
    
    if view_mode == "Current Status":
        years = sorted(df_yearly['Year'].unique())
        selected_year = st.sidebar.slider("📅 Year", int(min(years)), int(max(years)), int(max(years)))
    
    if view_mode == "🔮 AI Forecast (ML-Powered)":
        forecast_years = st.sidebar.slider("🔮 Forecast Horizon (Years)", 5, 20, 15)
        
        model_choice = st.sidebar.radio(
            "🤖 Prediction Model",
            ["Hybrid (Recommended)", "Ensemble Only", "XGBoost Only"]
        )
        
        show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if df_yearly is not None and df_monthly is not None:
    
    # ========================================================================
    # VIEW: AI FORECAST (ML-POWERED)
    # ========================================================================
    
    if view_mode == "🔮 AI Forecast (ML-Powered)":
        st.header(f"🔮 AI Forecast ({forecast_years} Years) - {selected_district}")
        
        district_history_yearly = df_yearly[df_yearly['District'] == selected_district].sort_values('Year')
        district_history_monthly = df_monthly[df_monthly['District'] == selected_district].sort_values(['Year', 'Month'])
        
        if len(district_history_yearly) >= 3 and len(district_history_monthly) >= 12:
            
            with st.spinner("🤖 Generating ML-powered predictions..."):
                if model_choice == "Hybrid (Recommended)":
                    forecast_df = hybrid_forecast_fixed(models, district_history_yearly, district_history_monthly, forecast_years)
                elif model_choice == "Ensemble Only":
                    forecast_df = forecast_with_actual_ensemble(models, district_history_yearly, district_history_monthly, forecast_years)
                else:
                    forecast_df = forecast_with_actual_xgboost(models, district_history_yearly, forecast_years)
            
            if forecast_df is not None:
                # Calculate statistics
                last_historical = district_history_yearly['Scarcity_Index'].iloc[-1]
                forecast_trend = "Increasing" if forecast_df['Scarcity_Index'].iloc[-1] > last_historical else "Decreasing"
                avg_change = (forecast_df['Scarcity_Index'].iloc[-1] - last_historical) / forecast_years
                
                # Show key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current (Latest)",
                        f"{last_historical:.2f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        f"Forecast ({int(forecast_df['Year'].iloc[-1])})",
                        f"{forecast_df['Scarcity_Index'].iloc[-1]:.2f}",
                        delta=f"{forecast_df['Scarcity_Index'].iloc[-1] - last_historical:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Trend Direction",
                        forecast_trend,
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "Avg Annual Change",
                        f"{avg_change:.3f}",
                        delta=None
                    )
                
                # Chart
                st.subheader("📈 Historical Data + ML Forecast")
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=district_history_yearly['Year'],
                    y=district_history_yearly['Scarcity_Index'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#1976d2', width=3),
                    marker=dict(size=8)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['Year'],
                    y=forecast_df['Scarcity_Index'],
                    mode='lines+markers',
                    name=f'{model_choice} Forecast',
                    line=dict(color='#f44336', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                # Confidence intervals (if enabled)
                if show_confidence and 'Confidence' in forecast_df.columns:
                    # Calculate upper/lower bounds based on confidence
                    std_factor = (100 - forecast_df['Confidence']) / 100
                    upper_bound = forecast_df['Scarcity_Index'] * (1 + std_factor * 0.5)
                    lower_bound = forecast_df['Scarcity_Index'] * (1 - std_factor * 0.5)
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Year'].tolist() + forecast_df['Year'].tolist()[::-1],
                        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(244, 67, 54, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval',
                        showlegend=True
                    ))
                
                # Threshold lines
                fig.add_hline(y=0.5, line_dash="dot", line_color="green", 
                             annotation_text="Low Threshold", annotation_position="right")
                fig.add_hline(y=1.0, line_dash="dot", line_color="orange",
                             annotation_text="Medium Threshold", annotation_position="right")
                fig.add_hline(y=1.5, line_dash="dot", line_color="red",
                             annotation_text="High Threshold", annotation_position="right")
                
                fig.update_layout(
                    title=f'Water Scarcity Forecast: {district_history_yearly["Year"].min()} → {forecast_df["Year"].max()}',
                    xaxis_title='Year',
                    yaxis_title='Scarcity Index',
                    height=550,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Alerts
                critical_years = forecast_df[forecast_df['Scarcity_Level'] == 'Critical']
                high_years = forecast_df[forecast_df['Scarcity_Level'] == 'High']
                
                if len(critical_years) > 0:
                    st.error(f"🚨 CRITICAL ALERT: Critical scarcity expected starting {int(critical_years['Year'].min())}")
                elif len(high_years) > 0:
                    st.warning(f"⚠️ HIGH ALERT: High scarcity expected starting {int(high_years['Year'].min())}")
                else:
                    st.success("✅ Good News: Scarcity levels expected to remain manageable")
                
                # Detailed forecast table
                st.subheader("📅 Year-by-Year Forecast Details")
                
                # Format the dataframe for display
                display_df = forecast_df.copy()
                display_df['Year'] = display_df['Year'].astype(int)
                display_df['Scarcity_Index'] = display_df['Scarcity_Index'].round(3)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Forecast Data (CSV)",
                    csv,
                    f"{selected_district}_forecast_{forecast_years}years.csv",
                    "text/csv",
                    key='download-csv'
                )
                
                # Analysis panels
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Forecast Distribution")
                    level_counts = forecast_df['Scarcity_Level'].value_counts()
                    
                    fig_pie = px.pie(
                        values=level_counts.values,
                        names=level_counts.index,
                        title=f'Scarcity Levels Over Next {forecast_years} Years',
                        color_discrete_map={
                            'Low': '#4caf50',
                            'Medium': '#ff9800',
                            'High': '#ff5722',
                            'Critical': '#f44336'
                        },
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Recommended Actions")
                    
                    # Find worst year
                    worst_year_idx = forecast_df['Scarcity_Index'].idxmax()
                    worst_year = forecast_df.loc[worst_year_idx]
                    level, emoji, _, alert_class = get_scarcity_info(worst_year['Scarcity_Index'])
                    
                    st.markdown(f'<div class="{alert_class}">', unsafe_allow_html=True)
                    st.markdown(f"### {emoji} Peak: {level} Scarcity")
                    st.markdown(f"**Expected Year:** {int(worst_year['Year'])}")
                    st.markdown(f"**Scarcity Index:** {worst_year['Scarcity_Index']:.2f}")
                    st.markdown(f"**Confidence:** {worst_year['Confidence']:.1f}%")
                    st.markdown("**Required Actions:**")
                    for rec in get_recommendations(level):
                        st.markdown(f"- {rec}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Trend analysis
                st.subheader("📈 Trend Analysis")
                
                # Calculate year-over-year changes
                forecast_df['YoY_Change'] = forecast_df['Scarcity_Index'].diff()
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(
                    x=forecast_df['Year'],
                    y=forecast_df['YoY_Change'],
                    marker_color=['red' if x > 0 else 'green' for x in forecast_df['YoY_Change']],
                    name='Year-over-Year Change'
                ))
                
                fig_trend.update_layout(
                    title='Projected Year-over-Year Changes in Scarcity',
                    xaxis_title='Year',
                    yaxis_title='Change in Scarcity Index',
                    height=350
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
            else:
                st.error("❌ Unable to generate forecast. Please check if models are loaded correctly.")
        else:
            st.warning("⚠️ Need at least 3 years of historical data (yearly) and 12 months (monthly) to generate reliable forecasts.")
    
    # ========================================================================
    # OTHER VIEWS
    # ========================================================================
    
    elif view_mode == "Current Status":
        st.header(f"🎯 Current Status - {selected_district}")
        
        current_data = df_yearly[
            (df_yearly['District'] == selected_district) & 
            (df_yearly['Year'] == selected_year)
        ]
        
        if len(current_data) > 0:
            row = current_data.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                level, emoji, color, _ = get_scarcity_info(row['Scarcity_Index'])
                st.markdown(f"### {emoji} Scarcity Level: {level}")
                st.metric("Scarcity Index", f"{row['Scarcity_Index']:.2f}")
            
            with col2:
                st.metric("Water Demand", f"{row.get('Water_Demand_MCM', 0):.2f} MCM")
                st.metric("Water Supply", f"{row.get('Water_Supply_MCM', 0):.2f} MCM")
            
            with col3:
                st.metric("Population", f"{row.get('Total_Population', 0):,.0f}")
                st.metric("GWL (meters)", f"{row.get('GWL_meters', 0):.2f}")
        else:
            st.info(f"No data available for {selected_year}")
    
    elif view_mode == "Historical Trends":
        st.header(f"📈 Historical Trends - {selected_district}")
        
        district_data = df_yearly[df_yearly['District'] == selected_district].sort_values('Year')
        
        if len(district_data) > 0:
            fig = px.line(
                district_data,
                x='Year',
                y='Scarcity_Index',
                title=f'Historical Scarcity Trend: {selected_district}',
                markers=True
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(district_data[['Year', 'Scarcity_Index', 'Scarcity_Level']], use_container_width=True)
        else:
            st.info("No historical data available")
    
    elif view_mode == "National Overview":
        st.header("🗺️ National Overview")
        
        # State-wise statistics
        state_stats = df_yearly.groupby('State').agg({
            'Scarcity_Index': 'mean',
            'District': 'nunique'
        }).round(2).reset_index()
        state_stats.columns = ['State', 'Avg Scarcity Index', 'Number of Districts']
        
        st.subheader("State-wise Average Scarcity")
        st.dataframe(state_stats.sort_values('Avg Scarcity Index', ascending=False), use_container_width=True)
        
        # Top 10 districts with highest scarcity
        latest_year = df_yearly['Year'].max()
        top_districts = df_yearly[df_yearly['Year'] == latest_year].nlargest(10, 'Scarcity_Index')
        
        st.subheader(f"Top 10 High-Risk Districts ({latest_year})")
        
        fig = px.bar(
            top_districts,
            x='Scarcity_Index',
            y='District',
            orientation='h',
            color='Scarcity_Index',
            color_continuous_scale='Reds',
            title='Districts with Highest Water Scarcity'
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Unable to load data. Please check file paths.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Water Scarcity Prediction System | "
    "Powered by Advanced Machine Learning (Ensemble + XGBoost) | Now Using Actual Trained Models</p>",
    unsafe_allow_html=True
)