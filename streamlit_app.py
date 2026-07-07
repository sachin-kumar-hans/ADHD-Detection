import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
from catboost import CatBoostClassifier
from PIL import Image

# --- PAGE SETUP ---
st.set_page_config(page_title="MCB-Net: ADHD Neural Diagnostic Portal", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size:26px !important; font-weight: bold; color: #4B9CD3; text-align: center; }
    .report-box { padding: 20px; border-radius: 10px; background-color: #1e1e1e; border: 1px solid #2d2d2d; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">MCB-Net</p>', unsafe_allow_html=True)
st.write("Upload a sMRI or fMRI slice")

# --- MODEL LOADING WITH CACHING ---
class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=(self.num_capsules, self.input_num_capsules,
                                        self.dim_capsule, self.input_dim_capsule),
                                 initializer='glorot_uniform', name='W')
        self.built = True
    def squash(self, vectors, axis=-1):
        s_sq_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
        scale = s_sq_norm / (1.0 + s_sq_norm) / tf.sqrt(s_sq_norm + tf.keras.backend.epsilon())
        return scale * vectors
    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expanded, [1, self.num_capsules, 1, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, -1)
        pred_caps = tf.map_fn(lambda x: tf.matmul(self.W, x), inputs_tiled)
        pred_caps = tf.squeeze(pred_caps, axis=-1)
        b = tf.zeros(shape=[tf.shape(inputs)[0], self.num_capsules, self.input_num_capsules])
        for r in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = self.squash(tf.reduce_sum(tf.expand_dims(c, -1) * pred_caps, axis=2))
            if r < self.routings - 1:
                b += tf.reduce_sum(pred_caps * tf.expand_dims(outputs, 2), axis=-1)
        return outputs
    def get_config(self):
        config = super(CapsuleLayer, self).get_config()
        config.update({"num_capsules": self.num_capsules, "dim_capsule": self.dim_capsule, "routings": self.routings})
        return config

@st.cache_resource
def load_pipeline():
    extractor = tf.keras.models.load_model('hybrid_capsule_extractor.h5', custom_objects={'CapsuleLayer': CapsuleLayer}, compile=False)
    selector = joblib.load('selector_fold_1.pkl')
    classifier = CatBoostClassifier()
    classifier.load_model('catboost_fold_1.cbm')
    return extractor, selector, classifier

try:
    extractor, selector, classifier = load_pipeline()
    CLASS_NAMES = ["ADHD Positive","ADHD Negative (Control)"]
except Exception as e:
    st.error(f"Error loading models. Ensure model weights are in the same folder. Details: {e}")
    st.stop()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an MRI image file...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Display Input Frame
    st.image(uploaded_file, caption="Uploaded Scan Slice", use_container_width=True)
    
    with st.spinner("Processing structural components..."):
        # Preprocessing Matrix
        img_resized = cv2.resize(img, (224, 224))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_resized)
        img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)
        img_norm = img_blur.astype(np.float32) / 255.0
        tensor = np.stack([img_norm] * 3, axis=-1)
        
        # Inference Pipeline
        features = extractor.predict(np.expand_dims(tensor, axis=0), verbose=0)
        selected_features = selector.transform(features)
        prob = classifier.predict_proba(selected_features)[0]
        predicted_idx = np.argmax(prob)
        
        # Alignment check
        if hasattr(classifier, "classes_"):
            pred_str = str(classifier.classes_[predicted_idx])
            if pred_str in ["0", "0.0"]: pred_str = CLASS_NAMES[0]
            elif pred_str in ["1", "1.0"]: pred_str = CLASS_NAMES[1]
        else:
            pred_str = CLASS_NAMES[predicted_idx]
            
        confidence = prob[predicted_idx] * 100
        
    # --- RESULT DISPLAY ---
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.write("**DIAGNOSTIC STATUS OUTCOME**")
    
    if "Positive" in pred_str:
        st.markdown(f'<h2 style="color:#FF4A4A; margin:5px 0;">{pred_str}</h2>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h2 style="color:#4AFF4A; margin:5px 0;">{pred_str}</h2>', unsafe_allow_html=True)
        
    st.write(f"Confidence Level: **{confidence:.2f}%**")
    st.markdown('</div>', unsafe_allow_html=True)