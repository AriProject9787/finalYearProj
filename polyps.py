import os
# MUST be the first thing in the script to fix the 'batch_shape' error
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import time
from collections import deque
from io import BytesIO

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="AI Polyp Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid #333;
    }
    .header-text {
        color: #ff4b4b;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256
CLASSIFIER_MODEL_PATH = "classifier.h5"
SEGMENTATION_MODEL_PATH = "segmentation.h5"

# --- MODEL LOADING WITH CACHING ---
@st.cache_resource
def load_models():
    try:
        # compile=False avoids errors with optimizer/input layer serialization
        classifier = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH, compile=False)
        seg_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH, compile=False)
        return classifier, seg_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# --- CORE LOGIC ---
def process_image(img, classifier, seg_model):
    # 1. Classification
    img_cls = cv2.resize(img, (IMG_SIZE_CLS, IMG_SIZE_CLS))
    img_cls_norm = img_cls.astype(np.float32) / 255.0
    prob = classifier.predict(np.expand_dims(img_cls_norm, axis=0), verbose=0)[0][0]
    
    is_polyp = prob >= 0.5
    confidence = prob if is_polyp else (1 - prob)
    
    mask = None
    severity = 0
    overlay = None
    
    if is_polyp:
        # 2. Segmentation
        img_seg = cv2.resize(img, (IMG_SIZE_SEG, IMG_SIZE_SEG))
        img_seg_norm = img_seg.astype(np.float32) / 255.0
        mask_pred = seg_model.predict(np.expand_dims(img_seg_norm, axis=0), verbose=0)[0]
        
        binary_mask = (mask_pred > 0.3).astype(np.uint8)
        severity = round((np.sum(binary_mask) / (IMG_SIZE_SEG * IMG_SIZE_SEG)) * 100, 2)
        
        # 3. Overlay
        overlay = img_seg.copy()
        # Handle cases where mask might be 2D or 3D
        m_sq = binary_mask.squeeze()
        overlay[m_sq == 1] = [0, 255, 0] # Green overlay
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        mask = (m_sq * 255).astype(np.uint8)
        
    return is_polyp, confidence, severity, mask, overlay

# --- UI LAYOUT ---
def main():
    st.sidebar.markdown("<h1 class='header-text'>🔬 Navigation</h1>", unsafe_allow_html=True)
    app_mode = st.sidebar.selectbox("Choose the Mode", ["About", "Image Upload", "Live Stream / Video"])
    
    classifier, seg_model = load_models()
    if not classifier or not seg_model:
        st.warning("Models could not be loaded. Please ensure classifier.h5 and segmentation.h5 are in the root directory.")
        return

    if app_mode == "About":
        st.title("AI-Powered Polyp Detection & Analysis")
        st.markdown("""
        Welcome to the **Next-Gen Gastrointestinal Analysis Portal**. This application utilizes deep learning models to:
        1. **Classify** endoscopic images for presence of polyps.
        2. **Segment** the exact polyp area using a U-Net architecture.
        3. **Quantify** severity based on tissue area coverage.
        """)
        st.info("Built with TensorFlow & Streamlit • Medical Analysis v1.0")

    elif app_mode == "Image Upload":
        st.title("Polyp Analysis: Image Upload")
        uploaded_file = st.file_uploader("Upload an endoscopy image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Input Image")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with st.spinner("Analyzing tissue samples..."):
                is_polyp, conf, severity, mask, overlay = process_image(img, classifier, seg_model)
                
                with col2:
                    st.subheader("Analysis Metrics")
                    status = "✅ POLYP DETECTED" if is_polyp else "❌ NO POLYP DETECTED"
                    color = "#ff4b4b" if is_polyp else "#00ff00"
                    
                    st.markdown(f"<h2 style='color:{color}; text-align:center;'>{status}</h2>", unsafe_allow_html=True)
                    
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown(f"<div class='metric-card'><h3>Confidence</h3><h2>{conf*100:.1f}%</h2></div>", unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(f"<div class='metric-card'><h3>Severity</h3><h2>{severity}%</h2></div>", unsafe_allow_html=True)
            
            if is_polyp:
                st.divider()
                st.subheader("Segmentation Results")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.image(mask, caption="Predicted Mask", use_container_width=True)
                with res_col2:
                    st.image(overlay, caption="Overlay Visualization", use_container_width=True)

    elif app_mode == "Live Stream / Video":
        st.title("Real-Time & Video Polyp Analysis")
        tab1, tab2 = st.tabs(["📁 Upload Video", "🎥 Live Webcam"])
        
        with tab1:
            uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov"])
            if uploaded_video:
                import tempfile
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                
                cap = cv2.VideoCapture(tfile.name)
                frame_placeholder = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Logic for frame processing
                    is_p, conf, sev, msk, ovl = process_image(frame, classifier, seg_model)
                    
                    display_f = ovl if is_p else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_f, use_container_width=True)
                    
                cap.release()
                os.unlink(tfile.name)

        with tab2:
            st.error("Note: Browser-based webcam requires 'streamlit-webrtc' for cloud deployment. cv2.VideoCapture(0) only works locally.")
            if st.toggle('Start Local Webcam (Debug Only)'):
                cap = cv2.VideoCapture(0)
                win = st.image([])
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()

if __name__ == "__main__":
    main()
