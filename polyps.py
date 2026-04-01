import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
import time
import requests

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="AI Polyp Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
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
CLASSIFIER_ID = "1nQmClRILftvZe8C82mLR6VsYNlodu9w9"
CLASSIFIER_MODEL_PATH = "classifier.h5"
SEGMENTATION_MODEL_PATH = "segmentation.h5"

# --- HELPER: GDrive Download ---
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    
    def get_confirm_token(response):
        # 1. Try to get from cookies
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        
        # 2. Try to get from HTML content (Virus Scan Warning page)
        import re
        content = response.text
        # Google Drive sometimes uses: confirm=XXX&id=YYY or "confirm":"XXX"
        # We'll use a more aggressive regex
        match = re.search(r'confirm=([0-9A-Za-z_-]+)', content)
        if not match:
            # Try searching for the input name="confirm" value="XXX"
            match = re.search(r'name="confirm" value="([0-9A-Za-z_-]+)"', content)
        if match:
            return match.group(1)
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Downloading model: {destination}...")
    
    CHUNK_SIZE = 32768
    downloaded = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress_perc = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress_perc)
    
    progress_bar.empty()
    status_text.empty()
    
    # Final Validation: Check HDF5 signature
    with open(destination, "rb") as f:
        header = f.read(8)
        if header != b"\x89HDF\r\n\x1a\n":
            os.remove(destination)
            raise ValueError("Invalid model file format. Please download manually.")

# --- MODEL LOADING WITH CACHING ---
@st.cache_resource
def load_models():
    try:
        # Check if classifier exists
        if not os.path.exists(CLASSIFIER_MODEL_PATH):
            st.info("🚀 Initial Setup: Classifier model missing.")
            
            # 1. Attempt Automated Download
            # We use a session state to prevent infinite loops if download fails repeatedly
            if "download_attempted" not in st.session_state:
                st.session_state.download_attempted = False
            
            if not st.session_state.download_attempted:
                with st.spinner("Attempting to download model from Google Drive..."):
                    try:
                        download_file_from_google_drive(CLASSIFIER_ID, CLASSIFIER_MODEL_PATH)
                        st.success("Download complete!")
                        st.rerun()
                    except Exception as e:
                        st.session_state.download_attempted = True
                        st.error(f"⚠️ Automated download failed: {e}")
            
            # 2. Fallback: Manual Upload
            st.markdown(f"""
            ### 📥 Manual Model Setup Required
            GitHub has a 100MB limit, so the large **classifier.h5** model must be added manually:
            1. [Click here to download classifier.h5](https://drive.google.com/uc?export=download&id={CLASSIFIER_ID})
            2. **Upload** the file below once downloaded.
            """)
            
            uploaded_file = st.file_uploader("Upload 'classifier.h5' directly", type=["h5"], key="model_uploader")
            if uploaded_file:
                with open(CLASSIFIER_MODEL_PATH, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("Model saved successfully! Reloading...")
                st.rerun()
            
            return None, None
        
        # Check if segmentation exists
        if not os.path.exists(SEGMENTATION_MODEL_PATH):
            st.error(f"Error: {SEGMENTATION_MODEL_PATH} not found.")
            return None, None

        classifier = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
        seg_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH)
        return classifier, seg_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        if os.path.exists(CLASSIFIER_MODEL_PATH):
            os.remove(CLASSIFIER_MODEL_PATH)
        return None, None

def remove_specular_reflection(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold for very bright spots (reflective glare)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Expand the mask slightly to ensure full coverage
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    
    # Inpaint using Telea's algorithm
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# --- CORE LOGIC ---
def process_image(img, classifier, seg_model, apply_removal=False):
    # Apply preprocessing if enabled
    if apply_removal:
        img = remove_specular_reflection(img)

    # 1. Classification
    img_cls = cv2.resize(img, (IMG_SIZE_CLS, IMG_SIZE_CLS))
    img_cls_norm = img_cls.astype(np.float32) / 255.0
    prob = classifier.predict(np.expand_dims(img_cls_norm, axis=0), verbose=0)[0][0]
    
    is_polyp = prob >= 0.3
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
        overlay[binary_mask.squeeze() == 1] = [0, 255, 0] # Green overlay
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        mask = (binary_mask.squeeze() * 255).astype(np.uint8)
        
    return is_polyp, confidence, severity, mask, overlay

def create_combined_result(img, mask, overlay):
    # Standardize all to 256x256
    s = 256
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (s, s))
    
    if mask is not None:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_res = cv2.resize(mask_rgb, (s, s))
    else:
        # Create a blank black mask if no polyp detected
        mask_res = np.zeros((s, s, 3), dtype=np.uint8)
        
    overlay_res = cv2.resize(overlay, (s, s)) if overlay is not None else img_res

    # Concatenate horizontally with white separators
    sep = np.ones((s, 5, 3), dtype=np.uint8) * 255
    combined = np.hstack([img_res, sep, mask_res, sep, overlay_res])
    
    # Add labels using OpenCV (optional but helps clarity)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 20), font, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, "Mask", (s + 15, 20), font, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, "Result", (2*s + 20, 20), font, 0.6, (255, 255, 255), 2)

    return Image.fromarray(combined)


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
        
        ### How to use:
        - Use the sidebar to switch between **Image Upload** and **Live Stream**.
        - Upload an endoscopy image or connect your device's camera.
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
                # Preprocessing Toggle
                apply_removal = st.checkbox("Apply Specular Reflection Removal", value=True, help="Removes bright white glare to improve detection.")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with st.spinner("Analyzing tissue samples..."):
                is_polyp, conf, severity, mask, overlay = process_image(img, classifier, seg_model, apply_removal)
                
                with col2:
                    st.subheader("Analysis Metrics")
                    m_col1, m_col2 = st.columns(2)
                    
                    status = "✅ POLYP DETECTED" if is_polyp else "❌ NO POLYP DETECTED"
                    color = "red" if is_polyp else "green"
                    
                    st.markdown(f"<h2 style='color:{color}; text-align:center;'>{status}</h2>", unsafe_allow_html=True)
                    
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
                
                # Download button
                combined_pil = create_combined_result(img, mask, overlay)
                from io import BytesIO
                buf = BytesIO()
                combined_pil.save(buf, format="PNG")
                st.download_button(label="Download Combined Analysis", data=buf.getvalue(), file_name="polyp_composite.png", mime="image/png")


    elif app_mode == "Live Stream / Video":
        st.title("Real-Time & Video Polyp Analysis")
        st.markdown("Analyze live camera feeds or upload pre-recorded endoscopic videos.")
        
        tab1, tab2 = st.tabs(["📁 Upload Video", "🎥 Live Webcam"])
        
        with tab1:
            uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_video is not None:
                # Initialize session state for stats to persist through button clicks
                if "video_stats" not in st.session_state or st.session_state.get("last_video") != uploaded_video.name:
                    st.session_state.video_stats = {
                        "polyp_frames": 0,
                        "max_conf": 0.0,
                        "processed_count": 0
                    }
                    st.session_state.last_video = uploaded_video.name

                import tempfile
                from collections import deque
                
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile_name = tfile.name
                tfile.close() # Close handle immediately so other processes can access it
                
                try:
                    cap = cv2.VideoCapture(tfile_name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.info(f"Video Loaded: {total_frames} frames detected.")
                    
                    # --- CONTROL PANEL ---
                    with st.expander("⚙️ Analysis Settings", expanded=True):
                        c1, c2 = st.columns(2)
                        frame_range = c1.slider("Select Frame Range", 0, total_frames, (0, total_frames))
                        skip_step = c2.select_slider("Analysis Speed / Skip Rate", options=[1, 2, 5, 10, 25, 50], value=2, 
                                                   help="Higher values skip more frames, speeding up discovery. '1' processes every frame.")
                        apply_removal = st.checkbox("Specular Reflection Removal (Experimental)", value=True)
                        start_frame, end_frame = frame_range

                    with st.expander("🎯 Analyze Particular Frame"):
                        col_f1, col_f2 = st.columns([3, 1])
                        target_f = col_f1.slider("Seek Frame", start_frame, end_frame, start_frame)
                        status_msg = st.empty()
                        if col_f2.button("Run Analysis", use_container_width=True):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                            ret, f_img = cap.read()
                            if ret:
                                with st.spinner("Processing..."):
                                    is_p, c, s, m, o = process_image(f_img, classifier, seg_model, apply_removal)
                                    st.image(o if is_p else cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB), caption=f"Analysis of Frame {target_f}", use_container_width=True)
                                    res_c1, res_c2 = st.columns(2)
                                    res_c1.metric("Status", "Polyp" if is_p else "Clear")
                                    res_c2.metric("Confidence", f"{c*100:.1f}%")
                                    
                                    # Download button for single frame
                                    combined_pil = create_combined_result(f_img, m, o)
                                    from io import BytesIO
                                    buf = BytesIO()
                                    combined_pil.save(buf, format="PNG")
                                    st.download_button(label="Download This Frame Analysis", data=buf.getvalue(), 
                                                       file_name=f"frame_{target_f}_analysis.png", mime="image/png", key=f"dl_{target_f}")
                            else:
                                st.error("Failed to read frame.")

                    st.markdown("### 📽️ Bulk Video Analysis")
                    c1, c2, c3 = st.columns(3)
                    is_paused = c1.toggle("Pause Analysis", value=False)
                    stop_btn = c2.button("Stop & Show Result", type="secondary")
                    
                    # Set start frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    current_f = start_frame
                    
                    # Progress bar
                    prog_bar = st.progress(0)
                    frame_text = st.empty()
                    
                    # Metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    conf_placeholder = m_col1.empty()
                    sev_placeholder = m_col2.empty()
                    verdict_placeholder = m_col3.empty()
                    
                    frame_placeholder = st.empty()
                    
                    # Analysis Stats (using session state to persist through stop button)
                    stats = st.session_state.video_stats

                    
                    # Accuracy Buffer (Temporal Smoothing)
                    prediction_buffer = deque(maxlen=5) 
                    
                    while cap.isOpened() and current_f <= end_frame:
                        if stop_btn:
                            break
                        
                        if is_paused:
                            time.sleep(0.5)
                            continue

                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        current_f += 1
                        
                        # Implementation of Speed/Skip
                        if skip_step > 1:
                            for _ in range(skip_step - 1):
                                cap.grab()
                                current_f += 1
                        
                        # Use unified process_image logic
                        is_detected, conf, severity, mask, overlay_vid = process_image(frame, classifier, seg_model, apply_removal)
                        stats["processed_count"] += 1
                        
                        if is_detected:
                            stats["polyp_frames"] += 1
                            stats["max_conf"] = max(stats["max_conf"], conf)
                            
                            display_frame = overlay_vid
                            conf_placeholder.markdown(f"<div class='metric-card'><h3>Confidence</h3><h2>{conf*100:.1f}%</h2></div>", unsafe_allow_html=True)
                            sev_placeholder.markdown(f"<div class='metric-card'><h3>Severity</h3><h2>{severity}%</h2></div>", unsafe_allow_html=True)
                            verdict_placeholder.markdown(f"<div class='metric-card'><h3 style='color:red;'>STATUS</h3><h2>POLYP</h2></div>", unsafe_allow_html=True)
                            # Add text on frame for visual confirmation
                            cv2.putText(display_frame, "POLYP DETECTED", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                        else:
                            display_frame = frame.copy()
                            conf_placeholder.markdown(f"<div class='metric-card'><h3>Confidence</h3><h2>{(1-conf)*100:.1f}%</h2></div>", unsafe_allow_html=True)
                            sev_placeholder.markdown(f"<div class='metric-card'><h3>Severity</h3><h2>0.0%</h2></div>", unsafe_allow_html=True)
                            verdict_placeholder.markdown(f"<div class='metric-card'><h3 style='color:green;'>STATUS</h3><h2>CLEAR</h2></div>", unsafe_allow_html=True)
                            cv2.putText(display_frame, "CLEAR", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                        
                        frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Normalized Progress considering start/end range
                        total_to_process = (end_frame - start_frame) if (end_frame > start_frame) else 1
                        progress = min((current_f - start_frame) / total_to_process, 1.0)
                        prog_bar.progress(progress)
                        frame_text.text(f"Processing Frame: {current_f} / {end_frame} (Speed: {skip_step}x)")

                    
                    cap.release()
                    time.sleep(0.1) # Small buffer for Windows file release
                finally:
                    if os.path.exists(tfile_name):
                        try:
                            os.unlink(tfile_name)
                        except:
                            pass # Still locked? Let OS cleanup late
                
                # --- FINAL REPORT ---
                st.divider()
                st.subheader("🏁 Final Analysis Report")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                final_verdict = "⚠️ POLYP DETECTED" if stats["polyp_frames"] > 5 else "✅ NO POLYP FOUND"
                verdict_color = "red" if stats["polyp_frames"] > 5 else "green"
                
                with res_col1:
                    st.markdown(f"**Final Verdict:** <span style='color:{verdict_color}; font-size:24px;'>{final_verdict}</span>", unsafe_allow_html=True)
                with res_col2:
                    st.write(f"**Total Frames Processed:** {stats['processed_count']}")
                with res_col3:
                    st.write(f"**Max Confidence Seen:** {stats['max_conf']*100:.1f}%")
                
                if stats["polyp_frames"] > 0:
                    st.warning(f"Detection confirmed in {stats['polyp_frames']} frames. Further clinical inspection is recommended.")
                else:
                    st.success("Analysis complete. No signs of polyps detected in the provided video sample.")

        with tab2:
            st.warning("Ensure your camera is connected and not used by another application.")
            run = st.toggle('Start Webcam Stream', value=False)
            FRAME_WINDOW = st.image([], use_container_width=True)
            
            if run:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not access camera.")
                
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Feed disconnected.")
                        break
                    
                    img_cls = cv2.resize(frame, (IMG_SIZE_CLS, IMG_SIZE_CLS))
                    img_cls_norm = img_cls.astype(np.float32) / 255.0
                    prob = classifier.predict(np.expand_dims(img_cls_norm, axis=0), verbose=0)[0][0]
                    
                    display_frame = frame.copy()
                    if prob >= 0.3:
                        img_seg = cv2.resize(frame, (IMG_SIZE_SEG, IMG_SIZE_SEG))
                        mask_pred = seg_model.predict(np.expand_dims(img_seg.astype(np.float32)/255.0, axis=0), verbose=0)[0]
                        binary_mask = (mask_pred > 0.4).astype(np.uint8)
                        
                        mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))
                        display_frame[mask_resized == 1] = [0, 255, 0]
                        cv2.putText(display_frame, f"POLYP DETECTED ({prob*100:.1f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        cv2.putText(display_frame, f"CLEAR ({ (1-prob)*100:.1f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                cap.release()

if __name__ == "__main__":
    main()
