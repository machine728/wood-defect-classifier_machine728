import streamlit as st
import numpy as np
import pickle
from PIL import Image
from skimage import color
from skimage.feature import graycomatrix, graycoprops

# ── Konfigurasi ──────────────────────────────────────────────
MODEL_PATH = "model/knn_model.pkl"
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def extract_glcm_features(img_gray):
    img_uint8 = (img_gray * 255).astype(np.uint8)
    glcm = graycomatrix(img_uint8, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    energy      = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return contrast, energy, homogeneity, correlation

# ── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="Wood Defect Classifier", page_icon="🪵")

st.title("🪵 Wood Defect Classifier")
st.markdown("Klasifikasi jenis cacat permukaan kayu menggunakan **GLCM + KNN**")
st.divider()

uploaded = st.file_uploader("Upload gambar kayu", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    col1, col2 = st.columns(2)

    # Tampil gambar
    img_pil = Image.open(uploaded).convert("RGB")
    with col1:
        st.subheader("Gambar Input")
        st.image(img_pil, use_container_width=True)

    # Proses
    img_array = np.array(img_pil) / 255.0
    img_gray  = color.rgb2gray(img_array)

    contrast, energy, homogeneity, correlation = extract_glcm_features(img_gray)

    with col2:
        st.subheader("Fitur GLCM")
        st.metric("Contrast",    f"{contrast:.4f}")
        st.metric("Energy",      f"{energy:.4f}")
        st.metric("Homogeneity", f"{homogeneity:.4f}")
        st.metric("Correlation", f"{correlation:.4f}")

    st.divider()

    # Prediksi
    try:
        data   = load_model()
        model  = data["model"]
        feat   = np.array([[contrast, energy, homogeneity, correlation]])
        pred   = model.predict(feat)[0]
        proba  = model.predict_proba(feat)[0]
        conf   = max(proba) * 100
        classes = data["classes"]

        st.subheader("Hasil Klasifikasi")

        color_map = {"normal": "green", "dark_knot": "red", "knot": "orange"}
        badge_color = color_map.get(pred.lower(), "blue")

        st.markdown(
            f"<h2 style='color:{badge_color}; text-align:center;'>"
            f"Jenis Cacat: {pred.upper()}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center; font-size:18px;'>"
            f"Confidence: <b>{conf:.1f}%</b></p>",
            unsafe_allow_html=True
        )

        # Bar probabilitas per kelas
        st.subheader("Probabilitas per Kelas")
        for cls, prob in zip(classes, proba):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

    except FileNotFoundError:
        st.warning("⚠️ Model belum ada. Jalankan `train.py` terlebih dahulu!")

st.divider()
st.caption("Pengolahan Citra Digital — Tugas Akhir Praktikum")