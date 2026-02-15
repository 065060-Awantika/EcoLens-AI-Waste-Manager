import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EcoLens: AI Waste Audit",
    page_icon="♻️",
    layout="centered"
)

# --- LOAD MODEL FUNCTION ---
@st.cache_resource
def load_model():
    # 'compile=False' fixes the "optimizer" errors from version mismatch
    model = tf.keras.models.load_model('ecolens_model.h5', compile=False)
    return model

# --- UI HEADER ---
st.title("♻️ EcoLens: Smart Waste Intelligence")
st.markdown("### AI-Powered Sustainability Audit Tool")

# --- LOAD THE BRAIN ---
try:
    with st.spinner('Loading AI Model...'):
        model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- CLASS LABELS ---
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Drop Waste Image Here...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Specimen", width=300)

    # 2. Preprocess
    # Resize to 224x224 (Model Requirement)
    image_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Normalize
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Predict
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    label = class_names[idx]
    confidence = np.max(prediction) * 100

    # 4. Show Results
    st.divider()
    st.header(f"Detected: {label}")
    st.progress(int(confidence))
    st.caption(f"AI Confidence: {confidence:.1f}%")

    # 5. Business Logic
    st.subheader("📋 Sustainability Audit")
    c1, c2 = st.columns(2)
    
    with c1:
        st.info("**Operational Action**")
        if label in ['Cardboard', 'Paper']:
            st.write("Route: **Pulping Mill**")
        elif label in ['Glass', 'Metal', 'Plastic']:
            st.write("Route: **MRF (Recycling Center)**")
        else:
            st.write("Route: **Incineration**")

    with c2:
        st.success("**Economic Value**")
        if label == 'Metal':
            st.write("💰 High ($1500/ton)")
        elif label == 'Plastic':
            st.write("💰 Moderate ($400/ton)")
        elif label == 'Cardboard':
            st.write("💰 Moderate ($110/ton)")
        else:
            st.write("📉 Cost (Tipping Fees)")
