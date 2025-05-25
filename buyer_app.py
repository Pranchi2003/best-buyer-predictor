import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

# Load saved model and encoders
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model, label_encoders, target_encoder = pickle.load(f)
    return model, label_encoders, target_encoder

model, encoders, target_enc = load_model()

buyers = ['AgroBazaar Pvt Ltd', 'Krishi Mandal', 'Sarkari Procurement Center',
          'FieldFresh Traders', 'GreenYield Buyer Group']

# Buyer contact info with contact person
buyer_info = {
    'AgroBazaar Pvt Ltd': {
        'contact': '+91-9876543210',
        'contact_person': 'Mr. Ramesh Mehta (Procurement Officer)'
    },
    'Krishi Mandal': {
        'contact': '+91-9654321870',
        'contact_person': 'Ms. Priya Yadav (Mandate Manager)'
    },
    'Sarkari Procurement Center': {
        'contact': '1800-111-222 (Toll-Free)',
        'contact_person': 'Mr. Anil Sharma (Govt Mandi Supervisor)'
    },
    'FieldFresh Traders': {
        'contact': '+91-9112233445',
        'contact_person': 'Mr. Aman Joshi (Field Buyer)'
    },
    'GreenYield Buyer Group': {
        'contact': '+91-9988776655',
        'contact_person': 'Ms. Kavita Patel (Agro Relations Head)'
    }
}

st.title("🌾 Connect With Buyer's 🌾")

with st.form("predict"):
    crop = st.text_input("Crop Type", "Wheat")
    qty = st.slider("Quantity (quintals)", 1, 100, 20)
    loc = st.text_input("Location", "Hisar")
    moisture = st.slider("Moisture Content (%)", 5.0, 20.0, 12.5)
    exp_price = st.number_input("Expected Price (₹)", 1000, 5000, 2000)
    buyer_type = st.selectbox("Preferred Buyer Type", ["Private", "Government"])
    urgency = st.slider("Time Urgency (days)", 1, 7, 3)
    season = st.selectbox("Season", ["Rabi", "Kharif"])
    past_buyer = st.selectbox("Historical Buyer Deal", buyers)

    if st.form_submit_button("Predict Best Buyer"):
        sample = {
            'Crop_Type': crop,
            'Quantity': qty,
            'Location': loc,
            'Moisture_Content': moisture,
            'Expected_Price': exp_price,
            'Preferred_Buyer_Type': buyer_type,
            'Time_Urgency': urgency,
            'Season': season,
            'Historical_Buyer_Deals': past_buyer
        }

        for buyer in buyers:
            key = buyer.replace(" ", "_").replace(".", "").replace(",", "")
            sample[f'Offered_Price_{key}'] = random.randint(1500, 2600)
            sample[f'Distance_{key}'] = random.randint(5, 100)
            sample[f'Rating_{key}'] = round(random.uniform(3.0, 5.0), 1)

        input_df = pd.DataFrame([sample])

        # Encode input
        for col in encoders:
            le = encoders[col]
            if input_df[col].iloc[0] in le.classes_:
                input_df[col] = le.transform(input_df[col])
            else:
                input_df[col] = le.transform([le.classes_[0]])

        pred_index = model.predict(input_df)[0]
        best_buyer = target_enc.inverse_transform([pred_index])[0]

        # Get contact info
        info = buyer_info.get(best_buyer, {})

        # Display result
        st.success(f"✅ Recommended Buyer: {best_buyer}")
        st.markdown(f"📞 **Contact:** {info.get('contact', 'N/A')}")
        st.markdown(f"👤 **Contact Person:** {info.get('contact_person', 'N/A')}")
