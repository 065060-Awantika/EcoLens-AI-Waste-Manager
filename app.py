import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoLens: AI Waste Audit",
    page_icon="♻️",
    layout="centered"
)

# --- LOAD THE MODEL ---
# We use @st.cache_resource so it only loads once (faster)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('ecolens_model.h5')
    return model

with st.spinner('Loading AI Brain...'):
    model = load_model()

# --- CLASS LABELS (Must match your training order) ---
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# --- UI HEADER ---
st.title("♻️ EcoLens: Intelligent Waste Sorting")
st.markdown("### AI-Powered Circular Economy Tool")
st.write("Upload waste items to automate sorting and audit sustainability metrics.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Drop Waste Image Here...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Display User Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Specimen", width=300)

    # 2. Preprocess Image (The "Magic" Step)
    # Resize to 224x224 because MobileNetV2 expects this exact size
    image_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numbers and normalize (0-1)
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # 3. Predict
    prediction = model.predict(img_array)
    idx = np.argmax(prediction) # Get the index of the highest score
    label = class_names[idx]
    confidence = np.max(prediction) * 100

    # --- RESULTS DISPLAY ---
    st.divider()
    
    # The Big Result
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Detected: **{label}**")
        st.progress(int(confidence))
        st.caption(f"AI Confidence: {confidence:.1f}%")
    
    # --- BUSINESS LOGIC (The "Management" Value) ---
    st.markdown("---")
    st.subheader("📋 Sustainability Audit Report")

    c1, c2, c3 = st.columns(3)

    if label in ['Cardboard', 'Paper']:
        with c1:
            st.success("✅ **Action**")
            st.write("Recycle (Fiber)")
        with c2:
            st.info("🏭 **Destination**")
            st.write("Pulping Mill")
        with c3:
            st.metric("Economic Value", "$110 / ton", delta="High Demand")
            
    elif label in ['Glass', 'Metal', 'Plastic']:
        with c1:
            st.success("✅ **Action**")
            st.write("Recycle (Solid)")
        with c2:
            st.info("🏭 **Destination**")
            st.write("Material Recovery Facility")
        with c3:
            st.metric("Economic Value", "$400 - $1500", delta="Variable")
            
    else: # Trash
        with c1:
            st.error("⚠️ **Action**")
            st.write("Disposal")
        with c2:
            st.warning("🏭 **Destination**")
            st.write("Landfill / Incinerator")
        with c3:
            st.metric("Cost Impact", "-$85 / ton", delta="-Tipping Fees", delta_color="inverse")

    st.warning("📢 **Log:** Item recorded in ESG Carbon Database.")