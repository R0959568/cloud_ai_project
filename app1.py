"""
UK Housing Price Predictor - Streamlit App
============================================
Team: [Your Group Name]
Model: LightGBM

This app predicts UK housing prices based on:
- Location (town, district, county)
- Property type
- Tenure type
- Year and month
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Page config
st.set_page_config(
    page_title="UK Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 UK Housing Price Predictor")
st.markdown("""
This tool predicts house prices in England & Wales based on property characteristics and location.
**Model:** LightGBM trained on 5.9M transactions (1995-2017)
""")

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained LightGBM model"""
    try:
        with open('models/lightgbm_housing.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Make sure 'models/lightgbm_housing.pkl' exists.")
        st.stop()

@st.cache_data
def load_data():
    """Load sample data to get valid categories"""
    try:
        df = pd.read_parquet('data/cleaned/housing_FULL_clean.parquet')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Make sure 'data/cleaned/housing_FULL_clean.parquet' exists.")
        st.stop()

# Load
with st.spinner("Loading model and data..."):
    model = load_model()
    df = load_data()

st.success("✅ Model loaded successfully!")

# Sidebar - Model Info
st.sidebar.header("📊 Model Information")
st.sidebar.metric("Model Type", "LightGBM")
st.sidebar.metric("Training Records", "5.9M")
st.sidebar.metric("R² Score", "~67%")
st.sidebar.metric("Years Covered", "1995-2017")

# Main form
st.header("🔮 Predict House Price")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    
    # Get unique values
    counties = sorted(df['county'].unique())
    county = st.selectbox("County", counties, index=counties.index('GREATER LONDON') if 'GREATER LONDON' in counties else 0)
    
    # Filter districts by county
    districts_in_county = sorted(df[df['county'] == county]['district'].unique())
    district = st.selectbox("District", districts_in_county)
    
    # Filter towns by district
    towns_in_district = sorted(df[df['district'] == district]['town_city'].unique())
    town_city = st.selectbox("Town/City", towns_in_district)

with col2:
    st.subheader("🏡 Property Details")
    
    property_types = sorted(df['property_type_label'].unique())
    property_type = st.selectbox("Property Type", property_types)
    
    tenure_types = sorted(df['tenure_label'].unique())
    tenure = st.selectbox("Tenure Type", tenure_types)
    
    is_new_build = st.checkbox("New Build", value=False)

with col3:
    st.subheader("📅 Date")
    
    year = st.slider("Year", 1995, 2017, 2017)
    month = st.slider("Month", 1, 12, 6)
    
    # Calculate quarter
    quarter = (month - 1) // 3 + 1
    st.info(f"Quarter: Q{quarter}")

# Predict button
st.markdown("---")
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    
    with st.spinner("Calculating price..."):
        
        # Prepare input data
        input_data = pd.DataFrame({
            'property_type_label': [property_type],
            'is_new_build': [int(is_new_build)],
            'tenure_label': [tenure],
            'county': [county],
            'district': [district],
            'town_city': [town_city],
            'year': [year],
            'month': [month],
            'quarter': [quarter]
        })
        
        # Convert categoricals to category dtype (LightGBM requirement)
        cat_columns = ['property_type_label', 'tenure_label', 'county', 'district', 'town_city']
        for col in cat_columns:
            input_data[col] = input_data[col].astype('category')
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success("✅ Prediction Complete!")
            
            # Big price display
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.markdown(f"""
                <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h1 style='color: #1f77b4; font-size: 48px; margin: 0;'>£{prediction:,.0f}</h1>
                    <p style='color: #666; margin-top: 10px;'>Estimated Property Value</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Price range (±10%)
            st.markdown("---")
            st.subheader("📊 Price Range Estimate")
            col_x, col_y, col_z = st.columns(3)
            
            lower_bound = prediction * 0.9
            upper_bound = prediction * 1.1
            
            col_x.metric("Lower Estimate (-10%)", f"£{lower_bound:,.0f}")
            col_y.metric("Predicted Price", f"£{prediction:,.0f}")
            col_z.metric("Upper Estimate (+10%)", f"£{upper_bound:,.0f}")
            
            # Summary
            st.markdown("---")
            st.subheader("📝 Property Summary")
            summary = f"""
            - **Location:** {town_city}, {district}, {county}
            - **Property Type:** {property_type}
            - **Tenure:** {tenure}
            - **New Build:** {'Yes' if is_new_build else 'No'}
            - **Date:** {year}-{month:02d} (Q{quarter})
            """
            st.markdown(summary)
            
            # Comparison with similar properties
            st.markdown("---")
            st.subheader("📈 Market Context")
            
            # Find similar properties in dataset
            similar = df[
                (df['town_city'] == town_city) &
                (df['property_type_label'] == property_type) &
                (df['year'] == year)
            ]
            
            if len(similar) > 0:
                avg_price = similar['price'].mean()
                median_price = similar['price'].median()
                
                col_i, col_ii, col_iii = st.columns(3)
                col_i.metric("Your Prediction", f"£{prediction:,.0f}")
                col_ii.metric("Average (Similar)", f"£{avg_price:,.0f}")
                col_iii.metric("Median (Similar)", f"£{median_price:,.0f}")
                
                st.info(f"ℹ️ Found {len(similar):,} similar properties in {town_city} ({year})")
            else:
                st.warning("⚠️ No similar properties found in dataset for comparison")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.error("Make sure the model was trained with the same features!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🏠 UK Housing Price Predictor | Built with Streamlit & LightGBM</p>
    <p>Data: England & Wales (1995-2017) | Model R²: ~67%</p>
</div>
""", unsafe_allow_html=True)
