# 🔬 AI-Powered Polyp Detection & Analysis Portal

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-polyp-detector-ariramaselvam.streamlit.app/)

**AI Polyp Detector** is a next-generation gastrointestinal analysis tool designed to assist medical professionals in the early detection and segmentation of colorectal polyps from endoscopic images and video streams. 

Using advanced Deep Learning models (Classification + Segmentation), this platform provides a professional-grade prototype for real-time tissue analysis and automated clinical reporting.

---

## 📽️ Live Demo
Experience the portal live: [https://ai-polyp-detector-ariramaselvam.streamlit.app/](https://ai-polyp-detector-ariramaselvam.streamlit.app/)

---

## 🚀 Key Features

### 1. Dual-Model Intelligence
- **Phase 1 (Classification):** Detects if a polyp is present in the tissue sample using a high-precision binary classifier.
- **Phase 2 (Segmentation):** Employs a U-Net architecture to precisely segment the polyp boundaries, highlighting areas of concern in a green overlay.

### 2. Advanced Video & Live Navigation
- **Analysis Range Control:** Use a frame-range slider to isolate specific sections of a video for targeted analysis.
- **Variable Skip Rate:** Speed up discovery by skipping frames (1x to 50x) while processing long endoscopy videos.
- **Single-Frame Seek & Analyze:** Seek any specific frame and run an immediate, detailed analysis (Mask + Overlay) without running the full video.

### 3. Smart Clinical Preprocessing
- **Specular Reflection Removal:** Implements an automated "Inpainting" algorithm (Telea) to neutralize bright white glare from the endoscope's light source, significantly reducing false positives and improving model focus on tissue morphology.

### 4. Professional Clinical Reporting
- **Composite Analysis Download:** Generates a 3-in-1 composite report image (Original | Mask | Final Result) with one click, providing clear, labeled documentation for medical records.

---

## 🛠️ Technology Stack
- **AI Core:** TensorFlow 2.15, Keras
- **Front-end / App Framework:** Streamlit
- **Image Processing:** OpenCV (cv2), Pillow (PIL)
- **State Management:** st.session_state (for persistent video analysis)

---

## 📂 Project Structure
- `polyps.py`: Main application logic, UI, and inference pipeline.
- `classifier.h5`: Deep learning model for polyp presence detection.
- `segmentation.h5`: U-Net model for pixel-level polyp segmentation.
- `requirements.txt`: Python dependencies.

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AriProject9787/finalYearProj
   cd finalYearProj
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run polyps.py
   ```

---

## 🎓 Academic Perspective
This project was developed for a **College Final Year Project**, demonstrating competencies in:
- **Medical AI & Computer Vision**
- **Deep Learning Model Integration**
- **Clinical Data Preprocessing (Glare Mitigation)**
- **End-to-End Software Architecture & UI/UX**

### Evaluation Metrics
- **Classification:** Precision, Recall (Sensitivity), and Accuracy.
- **Segmentation:** Dice Coefficient, Intersection over Union (IoU), and Pixel-wise Accuracy.

---

## ⚠️ Disclaimer
*This application is a research prototype intended for educational purposes only. It is NOT a substitute for clinical diagnosis by a certified medical professional.*

---

**Developed by Ariramaselvam • AI Medical Analysis v1.0**
