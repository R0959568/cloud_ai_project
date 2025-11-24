"""
UK Electricity Demand Predictor - Frontend
==========================================
Streamlit frontend that calls the FastAPI backend
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="UK Electricity Demand Predictor",
    page_icon="⚡",
    layout="wide"
)

# API endpoint - change based on deployment
# API_URL = "http://backend:8000"  # Docker: backend service name
API_URL = "http://localhost:8002"  # Local testing

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

# Check API health
def check_api_health():
    """Check if backend API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

# Make prediction via API
def predict_demand(datetime_str):
    """Call backend API to get prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"prediction_datetime": datetime_str},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API!")
        st.code(f"Backend URL: {API_URL}")
        st.info("Make sure backend is running: `cd backend && uvicorn api:app --reload`")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Get model info
def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Check API status
with st.spinner("Connecting to backend..."):
    api_healthy, health_data = check_api_health()

if api_healthy:
    st.success(f"✅ Connected to backend API")
    model_info = get_model_info()
else:
    st.error(f"❌ Backend API not responding")
    st.info("Start the backend: `cd backend && uvicorn api:app --reload`")
    model_info = None

# Sidebar - Model Info
st.sidebar.header("📊 Model Information")

if model_info:
    st.sidebar.metric("Model Type", "Gradient Boosting")
    st.sidebar.metric("Features", model_info.get('features_count', 39))
    
    perf = model_info.get('performance', {})
    st.sidebar.metric("R² Score", f"{perf.get('r2_score', 0):.4f}")
    st.sidebar.metric("MAE", f"{perf.get('mae_mw', 0):,.0f} MW")
    st.sidebar.metric("RMSE", f"{perf.get('rmse_mw', 0):,.0f} MW")
    
    hist_data = model_info.get('historical_data', {})
    if hist_data:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Historical Data**")
        st.sidebar.write(f"Records: {hist_data.get('total_records', 0):,}")
        
        with st.sidebar.expander("🔍 Feature Categories"):
            for category, features in model_info.get('feature_categories', {}).items():
                st.write(f"**{category.replace('_', ' ').title()}:**")
                st.write(f"- {len(features)} features")
else:
    st.sidebar.warning("⚠️ Backend not connected")

# Main content
st.header("🔮 Predict Electricity Demand")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Input Parameters")
    
    # Date and time inputs
    prediction_date = st.date_input(
        "Select Date",
        value=datetime.now().date(),
        min_value=datetime(2001, 1, 1),
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
    st.subheader("📊 Derived Features")
    
    # Extract features
    day_of_week = prediction_datetime.weekday()
    quarter = (prediction_datetime.month - 1) // 3 + 1
    is_weekend = day_of_week >= 5
    
    # Season
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season = season_map[prediction_datetime.month]
    
    # Display features
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Year:** {prediction_datetime.year}")
        st.write(f"**Month:** {prediction_datetime.month}")
        st.write(f"**Hour:** {prediction_datetime.hour}")
        st.write(f"**Quarter:** Q{quarter}")
    with col_b:
        st.write(f"**Day of Week:** {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week]}")
        st.write(f"**Season:** {season}")
        st.write(f"**Weekend:** {'Yes' if is_weekend else 'No'}")
        
        # Peak indicators
        is_peak_morning = 7 <= prediction_datetime.hour <= 9
        is_peak_evening = 17 <= prediction_datetime.hour <= 20
        if is_peak_morning:
            st.write("⚡ **Peak: Morning**")
        elif is_peak_evening:
            st.write("⚡ **Peak: Evening**")
        else:
            st.write("**Peak:** Off-peak")

# Predict button
st.markdown("---")

if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
    
    if not api_healthy:
        st.error("❌ Backend API is not running! Cannot make predictions.")
    else:
        with st.spinner("Calling backend API..."):
            
            # Convert datetime to ISO format
            datetime_str = prediction_datetime.isoformat()
            
            # Call API
            result = predict_demand(datetime_str)
            
            if result:
                prediction = result['predicted_demand_mw']
                features_used = result.get('features_used', {})
                
                # Display result
                st.success("✅ Prediction Complete!")
                
                # Big demand display
                col_center = st.columns([1, 2, 1])[1]
                with col_center:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h1 style='color: #2E86AB; font-size: 48px; margin: 0;'>{prediction:,.0f} MW</h1>
                        <p style='color: #666; margin-top: 10px;'>Predicted Electricity Demand</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction range (±5%)
                st.markdown("---")
                st.subheader("📊 Demand Range Estimate")
                
                lower = prediction * 0.95
                upper = prediction * 1.05
                
                col_x, col_y, col_z = st.columns(3)
                col_x.metric("Lower (-5%)", f"{lower:,.0f} MW")
                col_y.metric("Predicted", f"{prediction:,.0f} MW")
                col_z.metric("Upper (+5%)", f"{upper:,.0f} MW")
                
                # Key features used
                st.markdown("---")
                st.subheader("🔍 Key Features Used")
                
                col_feat1, col_feat2 = st.columns(2)
                
                with col_feat1:
                    st.write("**Temporal Features:**")
                    st.write(f"- Year: {features_used.get('year', 'N/A')}")
                    st.write(f"- Month: {features_used.get('month', 'N/A')}")
                    st.write(f"- Hour: {features_used.get('hour', 'N/A')}")
                    st.write(f"- Weekend: {'Yes' if features_used.get('is_weekend', False) else 'No'}")
                    st.write(f"- Season: {['Winter', 'Spring', 'Summer', 'Autumn'][features_used.get('season', 0)]}")
                
                with col_feat2:
                    st.write("**Historical Demand:**")
                    st.write(f"- Lag 1 day: {features_used.get('demand_lag_1d', 0):,.0f} MW")
                    st.write(f"- Rolling 24h mean: {features_used.get('rolling_mean_24h', 0):,.0f} MW")
                    st.write("- Total features: 39")
                
                # Context information
                st.markdown("---")
                st.subheader("💡 Demand Context")
                
                if prediction < 25000:
                    st.info("🌙 **Low Demand** - Typical for nighttime or off-peak hours")
                elif prediction < 35000:
                    st.info("📊 **Medium Demand** - Normal daytime usage")
                elif prediction < 42000:
                    st.warning("⚡ **High Demand** - Peak hours or high usage period")
                else:
                    st.error("🔥 **Very High Demand** - Extreme peak conditions")
                
                # Show request/response (for debugging)
                with st.expander("🔍 API Request/Response"):
                    col_req, col_res = st.columns(2)
                    with col_req:
                        st.write("**Request:**")
                        st.json({"prediction_datetime": datetime_str})
                    with col_res:
                        st.write("**Response:**")
                        st.json(result)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>⚡ UK Electricity Demand Predictor | Frontend: Streamlit | Backend: FastAPI + Gradient Boosting</p>
    <p>39 Enhanced Features | R² Score: 0.70 | Data: 2001-2025</p>
</div>
""", unsafe_allow_html=True)
