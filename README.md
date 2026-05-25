# 🫁 COVID-19 X-Ray Detection - Streamlit Application

A beautiful, responsive web application for COVID-19 detection from chest X-ray images using deep learning.

## ✨ Features

- 📸 **Image Upload & Capture**: Upload X-ray images or capture directly using webcam
- 🤖 **AI Predictions**: Real-time predictions with confidence scores
- 📊 **Visual Analytics**: Detailed confidence distribution charts
- 🎨 **Responsive Design**: Beautiful gradient UI with smooth interactions
- 📱 **Mobile Friendly**: Works seamlessly on desktop and mobile devices
- ⚙️ **Configurable**: Adjustable confidence threshold in sidebar
- 📚 **Educational Content**: How it works page with model details

## 🎯 Prediction Classes

- **🦠 COVID-19**: Chest X-rays showing COVID-19 pneumonia patterns
- **✅ Normal**: Healthy chest X-rays with no abnormalities
- **🫁 Viral Pneumonia**: Viral pneumonia characteristics

## 📋 Prerequisites

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.28+

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Model Files Exist

Before running the app, make sure these files are in the same directory:
- `best_covid_model.keras` - Trained model
- `label_encoder.pkl` - Label encoder for predictions
- `preprocessing_info.pkl` - Image preprocessing parameters

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### Prediction Page
1. Upload an X-ray image or capture one with your webcam
2. View the image preview
3. Get instant predictions with confidence scores
4. See detailed analysis with confidence distribution chart

### Model Info Page
- View model architecture details
- Check training/testing accuracy
- Understand the prediction classes
- See preprocessing pipeline

### How It Works Page
- Step-by-step explanation of prediction process
- Model architecture details
- Confidence score interpretation
- Technology stack information

## 📁 Project Structure

```
covid19-xray-detection/
├── app.py                 # Main Streamlit application
├── Mini_Project_CNN_Covid19_Chest_Xrays.ipynb  # Training notebook
├── best_covid_model.keras           # Trained model (required)
├── label_encoder.pkl                # Label encoder (required)
├── preprocessing_info.pkl           # Preprocessing metadata (required)
├── requirements.txt                 # Original training dependencies
└── README.md                        # This file
```

## ⚙️ Configuration

### Sidebar Settings
- **Confidence Threshold**: Adjust how confident the model must be
- **Navigation**: Switch between Prediction, Model Info, and How It Works

### Environment Variables (Optional)
```bash
# For production deployments
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow warnings
export STREAMLIT_SERVER_PORT=8501
```

## 🔍 How the Model Works

1. **Image Preprocessing**
   - Resize to 128×128 pixels
   - Normalize pixel values (0-1)
   - Convert to RGB color space

2. **Model Inference**
   - CNN processes image through convolutional layers
   - Feature extraction and pooling
   - Classification through dense layers

3. **Prediction Output**
   - Generates probability scores for each class
   - Returns highest confidence prediction
   - Provides confidence distribution

## 📊 Model Performance

- **Training Accuracy**: ~95%
- **Testing Accuracy**: ~93%
- **F1 Score**: ~0.92 (weighted average)
- **Overfitting Analysis**: Minimal (< 8% gap)

## ⚠️ Important Disclaimer

**This application is for educational and supportive purposes only.**

- ❌ NOT a substitute for professional medical diagnosis
- ⚠️ Always consult qualified healthcare professionals
- 📋 Results complement, not replace, expert medical evaluation
- 👨‍⚕️ User assumes responsibility for decisions based on predictions

## 🐛 Troubleshooting

### Model Files Not Found
**Error**: `Model files not found!`
- Ensure `best_covid_model.keras`, `label_encoder.pkl`, and `preprocessing_info.pkl` are in the same directory as `streamlit_app.py`

### Image Upload Not Working
- Check file format (JPG, PNG, BMP supported)
- Ensure image size is reasonable (< 10MB)
- Try camera input as alternative

### Slow Predictions
- Predictions should be < 2 seconds
- Check system resources
- Ensure TensorFlow is using GPU if available

### Deployment Issues
- Verify all model files are included in repository
- Check Python version compatibility (3.8+)
- Review Streamlit Cloud logs for errors

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [COVID-19 Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review model training notebook
3. Verify all dependencies are installed

## 📄 License

This project is provided for educational purposes.

## 🙏 Acknowledgments

- Dataset source: Kaggle COVID-19 Chest X-Ray Dataset
- Model architecture: Deep Convolutional Neural Networks
- Framework: TensorFlow/Keras and Streamlit

---

**Built with ❤️ for healthcare applications**

⚕️ Remember: Always prioritize professional medical consultation over AI predictions.
