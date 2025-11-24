"""
UK Housing Price Predictor - Frontend
======================================
Streamlit frontend that calls the FastAPI backend
"""

import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="UK Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# API endpoint
API_URL = "http://84.235.175.123:8000"

# Title
st.title("🏠 UK Housing Price Predictor")
st.markdown("""
This tool predicts house prices in England & Wales based on property characteristics.
**Model:** LightGBM trained on 5.9M transactions (1995-2017)
""")

# Check API health
def check_api_health():
    """Check if backend API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Make prediction via API
def predict_price(data):
    """Call backend API to get prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API!")
        st.code(f"Backend URL: {API_URL}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Check API status
with st.spinner("Checking backend..."):
    api_healthy = check_api_health()

if api_healthy:
    st.success(f"✅ Connected to backend API: {API_URL}")
else:
    st.error(f"❌ Backend API not responding: {API_URL}")

# Sidebar
st.sidebar.header("📊 System Status")
st.sidebar.metric("Backend API", "Online ✅" if api_healthy else "Offline ❌")
st.sidebar.metric("API URL", API_URL)
st.sidebar.markdown("---")
st.sidebar.metric("Model Type", "LightGBM")
st.sidebar.metric("R² Score", "~67%")

# Main form
st.header("🔮 Predict House Price")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    county = st.text_input("County", "GREATER LONDON")
    district = st.text_input("District", "CITY OF WESTMINSTER")
    town_city = st.text_input("Town/City", "LONDON")

with col2:
    st.subheader("🏡 Property Details")
    property_type = st.text_input("Property Type", "Detached")
    tenure = st.text_input("Tenure Type", "Freehold")
    is_new_build = st.checkbox("New Build", value=False)

with col3:
    st.subheader("📅 Date")
    year = st.slider("Year", 1995, 2017, 2017)
    month = st.slider("Month", 1, 12, 6)
    quarter = (month - 1) // 3 + 1
    st.info(f"Quarter: Q{quarter}")

# Predict button
st.markdown("---")

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    
    if not api_healthy:
        st.error("❌ Backend API is not running! Cannot make predictions.")
    else:
        with st.spinner("Calling backend API..."):
            
            # Prepare request data
            request_data = {
                "property_type_label": property_type,
                "is_new_build": is_new_build,
                "tenure_label": tenure,
                "county": county,
                "district": district,
                "town_city": town_city,
                "year": year,
                "month": month,
                "quarter": quarter
            }
            
            # Call API
            result = predict_price(request_data)
            
            if result:
                prediction = result['predicted_price']
                lower = result['lower_bound']
                upper = result['upper_bound']
                
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
                
                # Price range
                st.markdown("---")
                st.subheader("📊 Price Range Estimate")
                col_x, col_y, col_z = st.columns(3)
                
                col_x.metric("Lower (-10%)", f"£{lower:,.0f}")
                col_y.metric("Predicted", f"£{prediction:,.0f}")
                col_z.metric("Upper (+10%)", f"£{upper:,.0f}")
                
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
                
                # Show request/response (for debugging)
                with st.expander("🔍 API Request/Response"):
                    col_req, col_res = st.columns(2)
                    with col_req:
                        st.json(request_data)
                    with col_res:
                        st.json(result)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🏠 UK Housing Price Predictor | Frontend: Streamlit | Backend: FastAPI + LightGBM</p>
</div>
""", unsafe_allow_html=True)
