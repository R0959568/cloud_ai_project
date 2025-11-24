import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# Page config
st.set_page_config(page_title="UK House Price Predictor", page_icon="🏠", layout="wide")

# Title
st.title("🏠 UK House Price Predictor")
st.write("Predict house prices based on property features")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("This app predicts UK house prices using a CatBoost model trained on 496K properties.")

# Load model
@st.cache_resource
def load_prediction_model():
    return load_model('models/pycaret_best_model')

model = load_prediction_model()
st.success("✅ Model loaded successfully!")

# Input form
st.header("Enter Property Details:")

col1, col2, col3 = st.columns(3)

with col1:
    property_type = st.selectbox("Property Type", ['D', 'S', 'T', 'F', 'O'], 
                                  help="D=Detached, S=Semi-Detached, T=Terraced, F=Flat, O=Other")
    county = st.text_input("County", "GREATER LONDON")
    year = st.number_input("Year", min_value=1995, max_value=2025, value=2020)

with col2:
    is_new = st.selectbox("New Build?", ['N', 'Y'])
    district = st.text_input("District", "WESTMINSTER")
    month = st.slider("Month", 1, 12, 6)

with col3:
    tenure_type = st.selectbox("Tenure Type", ['F', 'L'], help="F=Freehold, L=Leasehold")
    town_city = st.text_input("Town/City", "LONDON")
    quarter = st.slider("Quarter", 1, 4, 2)

# Predict button
if st.button("🔮 Predict Price", type="primary"):
    # Create input dataframe
    # Create input dataframe with ALL required features
    input_data = pd.DataFrame({
        'property_type': [property_type],
        'is_new': [is_new],
        'tenure_type': [tenure_type],
        'county': [county.upper()],
        'district': [district.upper()],
        'town_city': [town_city.upper()],
        'year': [year],
        'month': [month],
        'quarter': [quarter],
        'day_of_week': [0],  # Add default day of week
        'is_new_build': [is_new == 'Y'],  # Add boolean version
        'property_type_label': [{'D': 'Detached', 'S': 'Semi-Detached', 'T': 'Terraced', 'F': 'Flat', 'O': 'Other'}[property_type]],  # Add label
        'tenure_label': [{'F': 'Freehold', 'L': 'Leasehold'}[tenure_type]]  # Add label
    })
        
    # Make prediction
    prediction = predict_model(model, data=input_data)
    predicted_price = prediction['prediction_label'].values[0]
    
    # Display result
    st.success(f"### Predicted Price: £{predicted_price:,.0f}")
    st.info(f"Typical error range: ±£19,681 (±32.5%)")
    st.write(f"**Estimated range:** £{predicted_price-19681:,.0f} - £{predicted_price+19681:,.0f}")