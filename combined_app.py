"""
Combined ML Prediction System - Unified Frontend
================================================
Single page with tabs for Housing and Electricity predictions
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="UK ML Prediction System",
    page_icon="🤖",
    layout="wide"
)

# API endpoints - Railway production URLs
HOUSING_API = os.getenv(
    "HOUSING_API_URL", 
    "https://friendly-clarity-housingandelectricity.up.railway.app"
)
ELECTRICITY_API = os.getenv(
    "ELECTRICITY_API_URL", 
    "https://housingelectricity-housingandelectricity.up.railway.app"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 UK ML Prediction System</h1>', unsafe_allow_html=True)
st.markdown("**Two powerful prediction models in one interface**")

# Check API health
@st.cache_data(ttl=60)
def check_apis():
    housing_ok = False
    electricity_ok = False
    try:
        housing_ok = requests.get(f"{HOUSING_API}/health", timeout=2).status_code == 200
    except:
        pass
    try:
        electricity_ok = requests.get(f"{ELECTRICITY_API}/health", timeout=2).status_code == 200
    except:
        pass
    return housing_ok, electricity_ok

housing_healthy, electricity_healthy = check_apis()

# Status indicators
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if housing_healthy:
        st.success("🏠 Housing: Online")
    else:
        st.error("🏠 Housing: Offline")
with col2:
    if electricity_healthy:
        st.success("⚡ Electricity: Online")
    else:
        st.error("⚡ Electricity: Offline")

st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["🏠 Housing Price Prediction", "⚡ Electricity Demand Prediction"])

# ============================================================================
# TAB 1: HOUSING PREDICTION
# ============================================================================
with tab1:
    st.header("UK Housing Price Prediction")
    st.markdown("Predict property prices based on location, type, and date (1995-2017 data)")
    
    if not housing_healthy:
        st.error("❌ Housing backend is not running. Start it with: `./start-housing.sh`")
    else:
        # Load data for dropdowns
        @st.cache_data
        def load_housing_data():
            paths = [
                'data/cleaned/housing_clean.parquet',
                '../data/cleaned/housing_clean.parquet',
                '../../data/cleaned/housing_clean.parquet',
            ]
            for path in paths:
                try:
                    return pd.read_parquet(path)
                except:
                    continue
            return None
        
        housing_df = load_housing_data()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Location")
            if housing_df is not None:
                counties = sorted(housing_df['county'].unique())
                county = st.selectbox("County", counties, 
                                     index=counties.index('GREATER LONDON') if 'GREATER LONDON' in counties else 0,
                                     key="housing_county")
                
                districts = sorted(housing_df[housing_df['county'] == county]['district'].unique())
                district = st.selectbox("District", districts, key="housing_district")
                
                towns = sorted(housing_df[housing_df['district'] == district]['town_city'].unique())
                town_city = st.selectbox("Town/City", towns, key="housing_town")
            else:
                county = st.text_input("County", "GREATER LONDON", key="housing_county_input")
                district = st.text_input("District", "CITY OF WESTMINSTER", key="housing_district_input")
                town_city = st.text_input("Town/City", "LONDON", key="housing_town_input")
        
        with col2:
            st.subheader("🏡 Property Details")
            if housing_df is not None:
                property_types = sorted(housing_df['property_type_label'].unique())
                property_type = st.selectbox("Property Type", property_types, key="housing_prop_type")
                
                tenure_types = sorted(housing_df['tenure_label'].unique())
                tenure = st.selectbox("Tenure Type", tenure_types, key="housing_tenure")
            else:
                property_type = st.selectbox("Property Type", 
                                            ["Detached", "Semi-Detached", "Terraced", "Flat", "Other"],
                                            key="housing_prop_type_default")
                tenure = st.selectbox("Tenure Type", ["Freehold", "Leasehold"], key="housing_tenure_default")
            
            is_new_build = st.checkbox("New Build", value=False, key="housing_new_build")
        
        with col3:
            st.subheader("📅 Date")
            year = st.slider("Year", 1995, 2017, 2017, key="housing_year")
            month = st.slider("Month", 1, 12, 6, key="housing_month")
            quarter = (month - 1) // 3 + 1
            st.info(f"Quarter: Q{quarter}")
        
        # Predict button
        if st.button("🔮 Predict House Price", type="primary", use_container_width=True, key="housing_predict"):
            with st.spinner("Calling API..."):
                try:
                    response = requests.post(
                        f"{HOUSING_API}/predict",
                        json={
                            "property_type_label": property_type,
                            "is_new_build": is_new_build,
                            "tenure_label": tenure,
                            "county": county,
                            "district": district,
                            "town_city": town_city,
                            "year": year,
                            "month": month,
                            "quarter": quarter
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result['predicted_price']
                        lower = result['lower_bound']
                        upper = result['upper_bound']
                        
                        st.success("✅ Prediction Complete!")
                        
                        # Display result
                        st.markdown(f"""
                        <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
                            <h1 style='color: #1f77b4; font-size: 48px; margin: 0;'>£{prediction:,.0f}</h1>
                            <p style='color: #666; margin-top: 10px;'>Estimated Property Value</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        col_x, col_y, col_z = st.columns(3)
                        col_x.metric("Lower (-10%)", f"£{lower:,.0f}")
                        col_y.metric("Predicted", f"£{prediction:,.0f}")
                        col_z.metric("Upper (+10%)", f"£{upper:,.0f}")
                        
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 2: ELECTRICITY PREDICTION
# ============================================================================
with tab2:
    st.header("UK Electricity Demand Prediction")
    st.markdown("Predict electricity demand using 39 advanced features (2001-2025 data)")
    
    if not electricity_healthy:
        st.error("❌ Electricity backend is not running. Start it with: `./start-electricity.sh`")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Input Parameters")
            
            prediction_date = st.date_input(
                "Select Date",
                value=datetime.now().date(),
                min_value=datetime(2001, 1, 1),
                max_value=datetime(2030, 12, 31),
                key="elec_date"
            )
            
            prediction_time = st.time_input(
                "Select Time",
                value=datetime.now().time(),
                key="elec_time"
            )
            
            prediction_datetime = datetime.combine(prediction_date, prediction_time)
            st.info(f"📅 Predicting for: **{prediction_datetime.strftime('%Y-%m-%d %H:%M')}**")
        
        with col2:
            st.subheader("📊 Derived Features")
            
            day_of_week = prediction_datetime.weekday()
            quarter = (prediction_datetime.month - 1) // 3 + 1
            is_weekend = day_of_week >= 5
            
            season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                         3: "Spring", 4: "Spring", 5: "Spring",
                         6: "Summer", 7: "Summer", 8: "Summer",
                         9: "Autumn", 10: "Autumn", 11: "Autumn"}
            season = season_map[prediction_datetime.month]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Year:** {prediction_datetime.year}")
                st.write(f"**Month:** {prediction_datetime.month}")
                st.write(f"**Hour:** {prediction_datetime.hour}")
                st.write(f"**Quarter:** Q{quarter}")
            with col_b:
                st.write(f"**Day:** {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week]}")
                st.write(f"**Season:** {season}")
                st.write(f"**Weekend:** {'Yes' if is_weekend else 'No'}")
                
                is_peak_morning = 7 <= prediction_datetime.hour <= 9
                is_peak_evening = 17 <= prediction_datetime.hour <= 20
                if is_peak_morning:
                    st.write("⚡ **Peak: Morning**")
                elif is_peak_evening:
                    st.write("⚡ **Peak: Evening**")
                else:
                    st.write("**Peak:** Off-peak")
        
        # Predict button
        if st.button("🔮 Predict Electricity Demand", type="primary", use_container_width=True, key="elec_predict"):
            with st.spinner("Calling API..."):
                try:
                    response = requests.post(
                        f"{ELECTRICITY_API}/predict",
                        json={"prediction_datetime": prediction_datetime.isoformat()},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result['predicted_demand_mw']
                        features_used = result.get('features_used', {})
                        
                        st.success("✅ Prediction Complete!")
                        
                        # Display result
                        st.markdown(f"""
                        <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
                            <h1 style='color: #2E86AB; font-size: 48px; margin: 0;'>{prediction:,.0f} MW</h1>
                            <p style='color: #666; margin-top: 10px;'>Predicted Electricity Demand</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        lower = prediction * 0.95
                        upper = prediction * 1.05
                        
                        col_x, col_y, col_z = st.columns(3)
                        col_x.metric("Lower (-5%)", f"{lower:,.0f} MW")
                        col_y.metric("Predicted", f"{prediction:,.0f} MW")
                        col_z.metric("Upper (+5%)", f"{upper:,.0f} MW")
                        
                        # Context
                        st.markdown("---")
                        if prediction < 25000:
                            st.info("🌙 **Low Demand** - Typical for nighttime or off-peak hours")
                        elif prediction < 35000:
                            st.info("📊 **Medium Demand** - Normal daytime usage")
                        elif prediction < 42000:
                            st.warning("⚡ **High Demand** - Peak hours or high usage period")
                        else:
                            st.error("🔥 **Very High Demand** - Extreme peak conditions")
                        
                        # Key features
                        with st.expander("🔍 Key Features Used"):
                            col_feat1, col_feat2 = st.columns(2)
                            with col_feat1:
                                st.write("**Temporal Features:**")
                                st.write(f"- Year: {features_used.get('year', 'N/A')}")
                                st.write(f"- Month: {features_used.get('month', 'N/A')}")
                                st.write(f"- Hour: {features_used.get('hour', 'N/A')}")
                                st.write(f"- Weekend: {'Yes' if features_used.get('is_weekend', False) else 'No'}")
                            with col_feat2:
                                st.write("**Historical Demand:**")
                                st.write(f"- Lag 1 day: {features_used.get('demand_lag_1d', 0):,.0f} MW")
                                st.write(f"- Rolling 24h: {features_used.get('rolling_mean_24h', 0):,.0f} MW")
                                st.write("- Total: 39 features")
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🏠 Housing Model**")
    st.caption("LightGBM | R²=0.67 | 994K records")
with col2:
    st.markdown("**⚡ Electricity Model**")
    st.caption("Gradient Boosting | R²=0.70 | 39 features")
with col3:
    st.markdown("**👥 Team Error400**")
    st.caption("Hamid Iqbal | Ibrahim Afkir")
