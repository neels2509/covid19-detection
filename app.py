import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# Configure Streamlit page
st.set_page_config(
    page_title="COVID-19 X-Ray Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Main content text */
.block-container {
    color: white;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0e1117;
}

/* Make ALL sidebar text white */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Header title */
.header-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}

/* Upload + camera containers */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
}

/* FIX upload button visibility */
[data-testid="stFileUploader"] section {
    background-color: white !important;
    border-radius: 10px;
    padding: 1rem;
}

/* Upload text */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span {
    color: black !important;
}

/* Browse files button */
[data-testid="stFileUploader"] button {
    background-color: #667eea !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}

/* Camera section */
[data-testid="stCameraInput"] button {
    background-color: #667eea !important;
    color: white !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] button {
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 10px 20px;
    margin-right: 5px;
    font-weight: 600;
    color: black;
}

.stTabs [aria-selected="true"] {
    background-color: #667eea !important;
    color: white !important;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}

/* Confidence gradients */
.confidence-high {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.confidence-medium {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.confidence-low {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
}

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.95);
    color: black;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.15);
}

/* Slider text */
.stSlider label {
    color: white !important;
}

/* Markdown text */
p, h1, h2, h3, h4, h5, h6 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Load model and preprocessing info
@st.cache_resource
def load_model_and_info():
    try:
        # model = load_model('best_covid_model.keras')
        # Download model once from hugging face (cached automatically)
        model_path = hf_hub_download(
            repo_id="neelshah259/covid-xray-model",
            filename="best_covid_model.keras",
            cache_dir=os.path.abspath(os.path.join(os.getcwd(), "models"))
        )
        model = load_model(model_path)
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        preprocessing_info = pickle.load(open('preprocessing_info.pkl', 'rb'))
        return model, label_encoder, preprocessing_info
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure the following files exist in the directory:")
        st.error("- best_covid_model.keras")
        st.error("- label_encoder.pkl")
        st.error("- preprocessing_info.pkl")
        st.stop()

model, label_encoder, preprocessing_info = load_model_and_info()

# Header
st.markdown('<h1 class="header-title">🫁 COVID-19 X-Ray Detection AI</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; color: white; font-size: 16px; margin-bottom: 2rem;">
    Advanced deep learning model for COVID-19 detection from chest X-ray images.
    Upload or capture an X-ray image to get instant predictions.
    </div>
    """, unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.markdown("### 📋 Navigation")
    page = st.radio("Select a page:", ["🔍 Prediction", "📊 Model Info", "❓ How It Works"])
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.markdown("### 📞 About")
    st.markdown("""
    **COVID-19 X-Ray Detection**
    
    Built with:
    - TensorFlow & Keras
    - Deep Neural Networks
    - Transfer Learning
    
    Classes: COVID-19, Normal, Viral Pneumonia
    """)

# ==================== PREDICTION PAGE ====================
if page == "🔍 Prediction":
    st.markdown("### 📸 Upload X-Ray Image")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option 1: Upload File**")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a chest X-ray image (JPG, PNG, BMP)"
        )
    
    with col2:
        st.markdown("**Option 2: Capture Photo**")
        camera_image = st.camera_input("Take a photo", help="Use your webcam to capture an image")
    
    # Important disclaimer
    st.info(
        "⚠️ **Important:** This model is designed specifically for **chest X-ray images only**. "
        "Predictions on non-medical images or images from other modalities may be unreliable. "
        "Always consult healthcare professionals for diagnosis."
    )
    
    # Determine which image to process
    image_to_process = None
    if uploaded_file is not None:
        image_to_process = uploaded_file
    elif camera_image is not None:
        image_to_process = camera_image
    
    if image_to_process is not None:
        # Display original image
        st.markdown("---")
        st.markdown("### 🖼️ Image Preview")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(image_to_process)
            st.image(image, use_column_width=True, caption="Uploaded X-Ray Image")
        
        # Preprocess image
        img_array = np.array(image.convert('RGB'))
        # Convert RGB to BGR to match training preprocessing (cv2.imread loads in BGR format)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        IMG_SIZE = preprocessing_info['IMG_SIZE']
        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        st.markdown("---")
        st.markdown("### 🤖 Generating Prediction...")
        
        with st.spinner("Analyzing X-ray image..."):
            predictions = model.predict(img_batch, verbose=0)[0]
        
        # Get predictions
        predicted_class_idx = np.argmax(predictions)
        predicted_class = label_encoder.classes_[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        max_confidence = np.max(predictions)
        
        # Check if prediction is reliable
        is_confident = max_confidence >= confidence_threshold
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        # Warn if confidence is too low or model is uncertain
        if max_confidence < 0.5:
            st.warning(
                "⚠️ **Low Confidence Alert**\n\n"
                "The model has low confidence in this prediction (< 50%). "
                "This image may not be a valid chest X-ray or contains unusual characteristics. "
                "Please verify the image quality and ensure it's a proper chest X-ray."
            )
        elif predicted_class == 'COVID-19' and max_confidence > 0.95:
            st.info(
                "ℹ️ **Very High Confidence Detected**\n\n"
                "This prediction has very high confidence. "
                "While this may indicate strong COVID-19 features, please still consult a healthcare professional for confirmation."
            )
        
        # Confidence gradient background
        if confidence >= 0.8:
            gradient = "confidence-high"
            status = "✅ High Confidence"
        elif confidence >= 0.6:
            gradient = "confidence-medium"
            status = "⚠️ Medium Confidence"
        else:
            gradient = "confidence-low"
            status = "❌ Low Confidence"
        
        # Main prediction card
        st.markdown(f"""
        <div class="prediction-card {gradient}">
            <h2 style="margin: 0; font-size: 2rem;">🎯 {predicted_class}</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">Confidence: {confidence:.2%}</p>
            <p style="font-size: 0.9rem; margin: 0;">{status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed predictions
        col1, col2, col3 = st.columns(3)
        
        for idx, class_name in enumerate(label_encoder.classes_):
            conf = predictions[idx]
            with col1 if idx == 0 else (col2 if idx == 1 else col3):
                st.markdown(f"""
                <div class="metric-box">
                    <h4>{class_name}</h4>
                    <h3>{conf:.2%}</h3>
                    <div style="background: #eee; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {conf*100}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence chart
        st.markdown("---")
        st.markdown("### 📈 Confidence Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#667eea' if i == predicted_class_idx else '#ccc' for i in range(len(predictions))]
        bars = ax.barh(label_encoder.classes_, predictions, color=colors)
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Confidence for Each Class', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, (bar, pred) in enumerate(zip(bars, predictions)):
            ax.text(pred + 0.02, i, f'{pred:.2%}', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Alert based on prediction
        st.markdown("---")
        if predicted_class == 'COVID-19':
            st.warning(
                "⚠️ **COVID-19 Detected**\n\n"
                "This X-ray shows characteristics consistent with COVID-19. "
                "Please consult with a healthcare professional for confirmation and treatment."
            )
        elif predicted_class == 'Viral Pneumonia':
            st.info(
                "ℹ️ **Viral Pneumonia Detected**\n\n"
                "This X-ray shows signs of viral pneumonia. "
                "Professional medical evaluation is recommended."
            )
        else:
            st.success(
                "✅ **Normal X-Ray**\n\n"
                "No significant abnormalities detected. "
                "However, always consult with a medical professional for accurate diagnosis."
            )
        
        # Model metadata
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h4>📐 Image Specifications</h4>
                <p><b>Input Size:</b> {IMG_SIZE}×{IMG_SIZE} pixels</p>
                <p><b>Normalization:</b> {preprocessing_info['normalization']}</p>
                <p><b>Channels:</b> 3 (BGR format)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h4>🧠 Model Details</h4>
                <p><b>Model Type:</b> Tuned CNN</p>
                <p><b>Classes:</b> {len(label_encoder.classes_)}</p>
                <p><b>Framework:</b> TensorFlow Keras</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== MODEL INFO PAGE ====================
elif page == "📊 Model Info":
    st.markdown("### 📚 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h4>🎯 Model Architecture</h4>
            <p><b>Type:</b> Convolutional Neural Network (CNN)</p>
            <p><b>Optimization:</b> Keras Tuner (Hyperband)</p>
            <p><b>Framework:</b> TensorFlow 2.x</p>
            <p><b>Input Shape:</b> 128×128×3</p>
            <p><b>Output Classes:</b> 3</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h4>🎓 Training Details</h4>
            <p><b>Dataset:</b> COVID-19 Chest X-Ray Dataset</p>
            <p><b>Train Accuracy:</b> ~95%</p>
            <p><b>Test Accuracy:</b> ~93%</p>
            <p><b>Optimizer:</b> Adam / RMSprop</p>
            <p><b>Loss Function:</b> Categorical Crossentropy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Classes Detected")
    
    classes_info = {
        '🦠 COVID-19': 'Chest X-rays showing pneumonia patterns consistent with COVID-19 infection',
        '✅ Normal': 'Healthy chest X-rays with no signs of infection or abnormality',
        '🫁 Viral Pneumonia': 'Chest X-rays showing viral pneumonia characteristics'
    }
    
    col1, col2, col3 = st.columns(3)
    for idx, (class_name, description) in enumerate(classes_info.items()):
        with col1 if idx == 0 else (col2 if idx == 1 else col3):
            st.markdown(f"""
            <div class="metric-box">
                <h4>{class_name}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔧 Preprocessing Pipeline")
    
    st.info("""
    ✓ Image resizing to 128×128 pixels
    ✓ Normalization (divide by 255.0)
    ✓ Color space conversion (RGB → BGR for model compatibility)
    ✓ Batch processing support
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Model Limitations")
    
    st.warning("""
    **This model is trained exclusively on chest X-ray images.**
    
    - ❌ Will NOT work reliably on CT scans, ultrasound, or other modalities
    - ❌ Will NOT work on non-medical images (screenshots, photos, etc.)
    - ❌ Predictions on out-of-distribution images may be meaningless
    - ⚠️ Always verify image is a proper chest X-ray before trusting results
    - 🩺 Always consult healthcare professionals for actual diagnosis
    
    **Low confidence scores (< 50%) likely indicate invalid input images.**
    """)

# ==================== HOW IT WORKS PAGE ====================
elif page == "❓ How It Works":
    st.markdown("### 🚀 How the COVID-19 Detection System Works")
    
    st.markdown("""
    ## Step-by-Step Process
    
    ### 1️⃣ Image Upload
    - User uploads a chest X-ray image (JPG, PNG, or BMP)
    - Image is validated and displayed for preview
    
    ### 2️⃣ Preprocessing
    - Image is resized to 128×128 pixels (standard model input)
    - Pixel values normalized (0-1 range by dividing by 255)
    - Color space converted to BGR (matches training data format from OpenCV)
    
    ### 3️⃣ Model Inference
    - Deep learning CNN processes the image
    - Analyzes patterns and features
    - Generates probability scores for each class
    
    ### 4️⃣ Prediction Output
    - Top prediction with confidence score
    - Confidence scores for all classes
    - Visual confidence distribution chart
    
    ### 5️⃣ Clinical Recommendation
    - AI provides interpretation
    - **Important:** Always consult healthcare professionals for diagnosis
    
    ---
    
    ## Model Architecture Details
    
    The model uses:
    - **Convolutional Layers** for feature extraction
    - **MaxPooling Layers** for dimensionality reduction
    - **Dense Layers** for classification
    - **Dropout** for regularization
    - **Hyperparameter Tuning** via Keras Tuner for optimization
    
    ---
    
    ## Confidence Score Interpretation
    
    | Confidence | Reliability |
    |---|---|
    | **> 80%** | High confidence (but verify image is a proper chest X-ray) |
    | **60-80%** | Moderate confidence - professional review recommended |
    | **< 60%** | Low confidence - image may not be a valid chest X-ray, or result is inconclusive |
    | **< 50%** | ⚠️ Very unreliable - image is likely not a chest X-ray |
    
    **Note:** Predictions are only valid for actual chest X-ray images. Non-medical images or images from other sources may produce meaningless results.
    
    ---
    
    ## Important Disclaimer
    
    ⚠️ **This AI tool is for educational and supportive purposes only.**
    
    - Results should NOT be used as sole basis for diagnosis
    - Always consult qualified healthcare professionals
    - The model predictions complement, not replace, expert medical evaluation
    - User assumes full responsibility for any decisions based on this tool
    
    ---
    
    ## Dataset Information
    
    - **Source:** COVID-19 Chest X-Ray Dataset
    - **Classes:** 3 (COVID-19, Normal, Viral Pneumonia)
    - **Training Set:** 80% of data
    - **Validation Set:** 10% of data
    - **Test Set:** 10% of data
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h4>🧬 ML Framework</h4>
            <ul style="list-style: none; padding: 0;">
                <li>✓ TensorFlow</li>
                <li>✓ Keras</li>
                <li>✓ OpenCV</li>
                <li>✓ NumPy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h4>🎨 Frontend</h4>
            <ul style="list-style: none; padding: 0;">
                <li>✓ Streamlit</li>
                <li>✓ Matplotlib</li>
                <li>✓ Seaborn</li>
                <li>✓ Pillow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h4>📊 Data Processing</h4>
            <ul style="list-style: none; padding: 0;">
                <li>✓ Scikit-learn</li>
                <li>✓ Pandas</li>
                <li>✓ Pickle</li>
                <li>✓ SciPy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px; padding: 2rem 0;">
    <p>🔬 COVID-19 X-Ray Detection AI | Powered by Deep Learning | Built with ❤️ for Healthcare</p>
    <p>⚠️ <b>Disclaimer:</b> This tool is for educational and supportive purposes only. Always consult with healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)
