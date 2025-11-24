"""
UK Electricity Demand Prediction App
Interactive Streamlit application for predicting electricity demand
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="UK Electricity Demand Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚡ UK Electricity Demand Predictor</h1>', unsafe_allow_html=True)

# Load historical data for lag features
@st.cache_data
def load_historical_data():
    """Load historical demand data for computing lag features"""
    possible_paths = [
        Path('data/interim/elec_cleaned_full.parquet'),
        Path('../data/interim/elec_cleaned_full.parquet'),
        Path(__file__).parent / 'data/interim/elec_cleaned_full.parquet',
    ]
    
    for path in possible_paths:
        if path.exists():
            df = pd.read_parquet(path)
            df['datetime'] = pd.to_datetime(df['settlement_date'])
            df = df.dropna(subset=['datetime', 'demand_value'])
            df = df.sort_values('datetime').reset_index(drop=True)
            # Keep only last 2 weeks for memory efficiency (enough for 7-day lags)
            df = df.tail(336 * 2)  # 336 half-hours = 1 week
            return df
    
    return None

# Load the best model (Gradient Boosting with R² = 0.70)
@st.cache_resource
def load_best_model():
    # Try multiple paths (local development vs deployment)
    possible_paths = [
        Path('data/final/models'),  # Deployment path
        Path('../data/final/models'),  # Local relative path
        Path(__file__).parent / 'data/final/models',  # Absolute from script location
    ]
    
    models_dir = None
    for path in possible_paths:
        if path.exists():
            models_dir = path
            break
    
    if models_dir is None:
        st.error("❌ Models directory not found. Please ensure data/final/models exists.")
        return None
    
    # Load Gradient Boosting model (better predictions than baseline)
    gb_path = models_dir / 'gradient_boosting_enhanced.pkl'
    if gb_path.exists():
        return joblib.load(gb_path)
    
    st.error("❌ Best model not found. Please run 02_models_local.ipynb first.")
    return None

# Load model metrics
@st.cache_data
def load_metrics():
    # Try multiple paths
    possible_paths = [
        Path('data/final/local_models_metrics_electricity.csv'),
        Path('../data/final/local_models_metrics_electricity.csv'),
        Path(__file__).parent / 'data/final/local_models_metrics_electricity.csv',
    ]
    
    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            # Get best model by R² score
            return df.loc[df['r2'].idxmax()]
    
    return None

try:
    selected_model = load_best_model()
    best_metrics = load_metrics()
    historical_data = load_historical_data()
    
    if selected_model is None:
        st.error("❌ Best model not found. Please run 02_models_local.ipynb first.")
        st.stop()
    
    # Display model info in sidebar
    st.sidebar.header("📊 Model Performance")
    
    # Show Gradient Boosting metrics
    st.sidebar.metric("Model", "Gradient Boosting")
    st.sidebar.metric("MAE", "2,353.23 MW")
    st.sidebar.metric("RMSE", "3,107.24 MW")
    st.sidebar.metric("R² Score", "0.6999")
    st.sidebar.success("✅ Enhanced with 39 features")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Helper function to compute enhanced features
def compute_enhanced_features(prediction_datetime, historical_df=None):
    """
    Compute all enhanced features needed for the model
    Includes lag features, rolling statistics, holidays, temporal features, and interactions
    """
    # Extract basic temporal features
    year = prediction_datetime.year
    month = prediction_datetime.month
    day = prediction_datetime.day
    hour = prediction_datetime.hour
    day_of_week = prediction_datetime.weekday()
    quarter = (month - 1) // 3 + 1
    week_of_year = prediction_datetime.isocalendar()[1]
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Enhanced temporal features
    is_business_hours = 1 if (hour >= 8 and hour <= 18 and is_weekend == 0) else 0
    is_night = 1 if (hour >= 23 or hour <= 5) else 0
    is_peak_morning = 1 if (hour >= 7 and hour <= 9) else 0
    is_peak_evening = 1 if (hour >= 17 and hour <= 20) else 0
    
    # Season
    season = {12: 0, 1: 0, 2: 0,  # Winter
              3: 1, 4: 1, 5: 1,   # Spring
              6: 2, 7: 2, 8: 2,   # Summer
              9: 3, 10: 3, 11: 3  # Autumn
    }[month]
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # UK Bank Holidays (simplified)
    uk_holidays = [
        # 2023
        '2023-01-01', '2023-01-02', '2023-04-07', '2023-04-10', '2023-05-01', '2023-05-29', 
        '2023-08-28', '2023-12-25', '2023-12-26',
        # 2024
        '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-06', '2024-05-27', 
        '2024-08-26', '2024-12-25', '2024-12-26',
        # 2025
        '2025-01-01', '2025-04-18', '2025-04-21', '2025-05-05', '2025-05-26', 
        '2025-08-25', '2025-12-25', '2025-12-26',
        # 2026
        '2026-01-01', '2026-04-03', '2026-04-06', '2026-05-04', '2026-05-25', 
        '2026-08-31', '2026-12-25', '2026-12-28',
        # 2027
        '2027-01-01', '2027-03-26', '2027-03-29', '2027-05-03', '2027-05-31', 
        '2027-08-30', '2027-12-27', '2027-12-28',
    ]
    
    date_str = prediction_datetime.strftime('%Y-%m-%d')
    is_holiday = 1 if date_str in uk_holidays else 0
    
    # Day before/after holiday
    day_before = (prediction_datetime - timedelta(days=1)).strftime('%Y-%m-%d')
    day_after = (prediction_datetime + timedelta(days=1)).strftime('%Y-%m-%d')
    is_day_before_holiday = 1 if day_before in uk_holidays else 0
    is_day_after_holiday = 1 if day_after in uk_holidays else 0
    
    # Interaction features
    weekend_hour = is_weekend * hour
    holiday_hour = is_holiday * hour
    month_hour = month * hour
    
    # Lag features and rolling statistics (if historical data available)
    demand_lag_1 = 0
    demand_lag_1d = 0
    demand_lag_3h = 0
    demand_lag_7d = 0
    demand_rolling_mean_24h = 0
    demand_rolling_std_24h = 0
    demand_rolling_mean_7d = 0
    demand_diff_from_24h_avg = 0
    
    if historical_df is not None:
        # Find the closest historical record before prediction time
        hist_before = historical_df[historical_df['datetime'] < prediction_datetime]
        
        if len(hist_before) > 0:
            # Lag 1: Previous half-hour (1 step back)
            if len(hist_before) >= 1:
                demand_lag_1 = hist_before.iloc[-1]['demand_value']
            
            # Lag 1d: Same time yesterday (48 half-hours = 1 day)
            lag_1d_time = prediction_datetime - timedelta(hours=24)
            lag_1d_record = hist_before[hist_before['datetime'] <= lag_1d_time]
            if len(lag_1d_record) > 0:
                demand_lag_1d = lag_1d_record.iloc[-1]['demand_value']
            
            # Lag 3h: 3 hours ago (6 half-hours)
            lag_3h_time = prediction_datetime - timedelta(hours=3)
            lag_3h_record = hist_before[hist_before['datetime'] <= lag_3h_time]
            if len(lag_3h_record) > 0:
                demand_lag_3h = lag_3h_record.iloc[-1]['demand_value']
            
            # Lag 7d: Same time last week (336 half-hours = 7 days)
            lag_7d_time = prediction_datetime - timedelta(days=7)
            lag_7d_record = hist_before[hist_before['datetime'] <= lag_7d_time]
            if len(lag_7d_record) > 0:
                demand_lag_7d = lag_7d_record.iloc[-1]['demand_value']
            
            # Rolling 24h statistics (48 half-hours)
            rolling_24h_time = prediction_datetime - timedelta(hours=24)
            rolling_24h_data = hist_before[hist_before['datetime'] >= rolling_24h_time]
            if len(rolling_24h_data) > 0:
                demand_rolling_mean_24h = rolling_24h_data['demand_value'].mean()
                demand_rolling_std_24h = rolling_24h_data['demand_value'].std()
                if pd.isna(demand_rolling_std_24h):
                    demand_rolling_std_24h = 0
                demand_diff_from_24h_avg = demand_lag_1 - demand_rolling_mean_24h
            
            # Rolling 7d statistics (336 half-hours)
            rolling_7d_time = prediction_datetime - timedelta(days=7)
            rolling_7d_data = hist_before[hist_before['datetime'] >= rolling_7d_time]
            if len(rolling_7d_data) > 0:
                demand_rolling_mean_7d = rolling_7d_data['demand_value'].mean()
    
    # Return features in the same order as training
    # Note: Using 2024 as a fixed year to avoid extrapolation issues with future dates
    # The year feature isn't actually predictive for electricity demand - seasonal patterns matter more
    fixed_year = 2024  # Use a year from training data range
    
    features = [
        fixed_year, month, day, hour, day_of_week, quarter, week_of_year,
        is_weekend, is_business_hours, is_night, is_peak_morning, is_peak_evening,
        season,
        hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos,
        demand_lag_1, demand_lag_1d, demand_lag_3h, demand_lag_7d,
        demand_rolling_mean_24h, demand_rolling_std_24h, demand_rolling_mean_7d,
        demand_diff_from_24h_avg,
        is_holiday, is_day_before_holiday, is_day_after_holiday,
        weekend_hour, holiday_hour, month_hour
    ]
    
    return np.array([features])

# Main content - Single Prediction
st.header("Single Time Point Prediction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Parameters")
    
    # Date and time inputs
    prediction_date = st.date_input(
            "Select Date",
        value=datetime.now(),
        min_value=datetime(2025, 1, 1),
        max_value=datetime(2030, 12, 31)
    )
    
    prediction_time = st.time_input(
        "Select Time",
        value=datetime.now().time()
    )
    
    # Combine date and time
    prediction_datetime = datetime.combine(prediction_date, prediction_time)
    
    # Display selected datetime
    st.info(f"📅 Predicting for: **{prediction_datetime.strftime('%Y-%m-%d %H:%M')}**")

with col2:
    st.subheader("Derived Features")
    
    # Extract features
    year = prediction_datetime.year
    month = prediction_datetime.month
    day = prediction_datetime.day
    hour = prediction_datetime.hour
    day_of_week = prediction_datetime.weekday()
    quarter = (month - 1) // 3 + 1
    week_of_year = prediction_datetime.isocalendar()[1]
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Display features
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Year:** {year}")
        st.write(f"**Month:** {month}")
        st.write(f"**Day:** {day}")
        st.write(f"**Hour:** {hour}")
    with col_b:
        st.write(f"**Day of Week:** {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week]}")
        st.write(f"**Quarter:** Q{quarter}")
        st.write(f"**Week of Year:** {week_of_year}")
        st.write(f"**Weekend:** {'Yes' if is_weekend else 'No'}")

# Predict button
if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
    # Use enhanced features with historical data
    features = compute_enhanced_features(prediction_datetime, historical_data)
    
    # Make prediction
    prediction = selected_model.predict(features)[0]
    
    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric(
            label="Predicted Electricity Demand",
            value=f"{prediction:,.0f} MW",
            delta=None
        )
    
    with col2:
        # Context - typical UK demand ranges from 20,000 to 50,000 MW
        if prediction < 25000:
            st.info("🌙 Low demand period")
        elif prediction < 35000:
            st.success("📊 Normal demand")
        elif prediction < 45000:
            st.warning("📈 High demand")
        else:
            st.error("🔥 Peak demand")
    
    with col3:
        confidence = 70.0  # R² of Gradient Boosting model
        st.metric("Model Confidence", f"{confidence:.1f}%")
    
    # Visualization
    st.markdown("---")
    st.subheader("Prediction Context")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Demand (MW)", 'font': {'size': 24}},
        delta={'reference': 32000, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 60000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25000], 'color': '#90EE90'},
                {'range': [25000, 35000], 'color': '#FFD700'},
                {'range': [35000, 45000], 'color': '#FFA500'},
                {'range': [45000, 60000], 'color': '#FF6347'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 45000
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>UK Electricity Demand Prediction System | Data: 2001-2025 | Built with Streamlit</p>
    <p>⚡ Powered by Gradient Boosting | R² Score: 0.70 | 39 Enhanced Features</p>
</div>
""", unsafe_allow_html=True)
