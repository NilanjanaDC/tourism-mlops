"""
Streamlit Web App for Tourism Wellness Package Prediction
Author: MLOps Team
Description: This app predicts whether a customer will purchase the wellness tourism package
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

# =============== Configuration ===============
st.set_page_config(
    page_title="Tourism Wellness Package Predictor",
    page_icon="✈️",
    layout="wide"
)

# HF model repo configuration (update with your repo)
HF_MODEL_REPO_ID = "your-username/tourism-wellness-model"
MODEL_FILENAME = "best_tourism_model.joblib"

# =============== Load Model ===============
@st.cache_resource
def load_model():
    """Load the trained model from Hugging Face"""
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=MODEL_FILENAME,
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# =============== App Title and Description ===============
st.title("✈️ Tourism Wellness Package Purchase Predictor")
st.markdown("---")
st.write(
    "This application uses machine learning to predict whether a customer "
    "will purchase the **Wellness Tourism Package**. Fill in the customer details "
    "below and click 'Predict' to get the prediction."
)
st.markdown("---")

# =============== Load Model ===============
model = load_model()

if model is not None:
    # =============== Create Input Form ===============
    st.header("📋 Customer Information Form")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=10000.0, max_value=500000.0, value=50000.0, step=5000.0)
    
    with col2:
        st.subheader("Trip Details")
        typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
        num_children = st.number_input("Number of Children Visiting (age < 5)", min_value=0, max_value=10, value=0, step=1)
        num_trips = st.number_input("Average Number of Trips per Year", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Sales Pitch Details")
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=120.0, value=15.0, step=1.0)
        num_followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=30.0, value=3.0, step=1.0)
        pitch_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
    
    with col4:
        st.subheader("Product & Preferences")
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        preferred_star = st.slider("Preferred Property Star (1-5)", 1, 5, 3)
        passport = st.selectbox("Passport Available", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.markdown("---")
    
    # =============== Make Prediction ===============
    if st.button("🔮 Predict Purchase", key="predict_button", use_container_width=True):
        # Create input dataframe with exact column names
        input_data = pd.DataFrame([{
            "Age": age,
            "TypeofContact": typeof_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_person_visiting,
            "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": num_children,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }])
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.success("✅ **WILL PURCHASE**", icon="🎉")
                    st.metric("Prediction", "Yes (1)", delta="High Probability")
                else:
                    st.warning("❌ **WILL NOT PURCHASE**", icon="📊")
                    st.metric("Prediction", "No (0)", delta="Low Probability")
            
            with col_result2:
                st.metric("Confidence Score", f"{max(probability)*100:.2f}%")
                
                # Probability breakdown
                st.write("**Probability Distribution:**")
                prob_df = pd.DataFrame({
                    "Outcome": ["No Purchase (0)", "Purchase (1)"],
                    "Probability": [probability[0]*100, probability[1]*100]
                })
                st.bar_chart(prob_df.set_index("Outcome"))
            
            st.markdown("---")
            st.info(
                "💡 **Interpretation:** This prediction is based on machine learning models "
                "(Random Forest & XGBoost) trained on historical customer data. "
                "Use this as one factor among others for decision-making."
            )
            
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")
            st.info("Please ensure all input values are valid and match the expected data types.")

else:
    st.error("❌ Failed to load the machine learning model. Please check the model repository configuration.")

# =============== Footer ===============
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Tourism Wellness Package Predictor | Built with Streamlit & MLOps | v1.0</p>
    <p>For support, contact: <a href='mailto:support@visitwithus.com'>support@visitwithus.com</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
