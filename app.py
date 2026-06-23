"""
Skin Doctor — AI-powered skin disease classifier
Streamlit app: upload a photo, get a diagnosis with confidence breakdown
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import os

# ── PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Doctor AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a5276;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #5d6d7e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #7d6608;
        margin: 1rem 0;
    }
    .result-card {
        background: #eafaf1;
        border: 1.5px solid #27ae60;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e8449;
    }
    .confidence-bar-label {
        font-size: 0.85rem;
        color: #2c3e50;
    }
    .low-confidence {
        background: #fdedec;
        border: 1.5px solid #e74c3c;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #922b21;
        margin: 0.5rem 0;
    }
    .info-section {
        background: #ebf5fb;
        border-left: 4px solid #2e86c1;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    div[data-testid="stProgress"] > div > div {
        background-color: #27ae60 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── DISEASE INFO DATABASE ─────────────────────────────────────────────────
DISEASE_INFO = {
    "Acne": {
        "description": "A skin condition causing pimples, blackheads, and whiteheads, usually on the face, chest, or back.",
        "causes": "Excess oil, clogged pores, bacteria (C. acnes), hormonal changes.",
        "treatment": "Topical retinoids, benzoyl peroxide, salicylic acid. Severe cases: antibiotics or isotretinoin.",
        "when_to_see_doctor": "If over-the-counter treatments don't help after 3 months, or if acne is severe and causing scarring.",
        "emoji": "🔴"
    },
    "Eczema": {
        "description": "A chronic inflammatory skin condition causing itchy, dry, and inflamed patches.",
        "causes": "Immune system overreaction, genetic factors, environmental triggers (soaps, fabrics, stress).",
        "treatment": "Moisturizers, topical corticosteroids, antihistamines, avoiding triggers.",
        "when_to_see_doctor": "If rash is widespread, infected (yellow crust, warmth), or not responding to moisturizers.",
        "emoji": "🟠"
    },
    "Psoriasis": {
        "description": "An autoimmune condition causing rapid skin cell buildup, resulting in scaling, red patches.",
        "causes": "Immune system triggering rapid skin cell growth. Stress, infections, and some medications can trigger flares.",
        "treatment": "Topical corticosteroids, vitamin D analogs, phototherapy, biologics for severe cases.",
        "when_to_see_doctor": "Always consult a dermatologist — psoriasis requires long-term management.",
        "emoji": "🟡"
    },
    "Rosacea": {
        "description": "A chronic condition causing facial redness, visible blood vessels, and sometimes small red bumps, mainly on the cheeks, nose, and forehead.",
        "causes": "Exact cause unknown. Linked to blood vessel abnormalities, Demodex mites, genetics, and triggers like sun exposure, alcohol, spicy food, and stress.",
        "treatment": "Topical metronidazole or azelaic acid, oral antibiotics for inflammation, laser therapy for visible vessels, avoiding personal triggers.",
        "when_to_see_doctor": "If redness persists, worsens, or is accompanied by eye irritation or thickening skin (especially around the nose).",
        "emoji": "🌸"
    },
    "Actinic Keratosis": {
        "description": "A rough, scaly patch on the skin caused by years of sun exposure. Considered a precancerous lesion.",
        "causes": "Cumulative UV damage from sun exposure over many years. More common in fair-skinned, older individuals.",
        "treatment": "Cryotherapy (freezing), topical 5-fluorouracil or imiquimod, photodynamic therapy, or curettage.",
        "when_to_see_doctor": "Always — these lesions can progress to squamous cell carcinoma if untreated, so prompt evaluation is important.",
        "emoji": "🟤"
    },
    "Basal Cell Carcinoma": {
        "description": "The most common type of skin cancer, typically appearing as a pearly bump, flat scaly patch, or sore that doesn't heal.",
        "causes": "Long-term UV exposure (sun or tanning beds) damaging skin cell DNA. Risk increases with fair skin, age, and history of sunburns.",
        "treatment": "Surgical excision, Mohs surgery, cryotherapy, topical chemotherapy, or radiation depending on size and location.",
        "when_to_see_doctor": "Immediately — any new or changing growth, sore that won't heal, or unusual mole-like lesion needs prompt dermatologist evaluation.",
        "emoji": "⚠️"},
    "Healthy_Skin": {
        "description": "No significant skin disease detected. Skin appears healthy.",
        "causes": "N/A",
        "treatment": "Maintain a daily skincare routine: gentle cleansing, moisturizing, and SPF 30+ sunscreen.",
        "when_to_see_doctor": "Annual skin check is recommended, especially if you have a family history of skin cancer.",
        "emoji": "✅"
    },
}

# ── MODEL LOADING ─────────────────────────────────────────────────────────
MODEL_PATH  = "best_model_phase2.h5"         # adjust path as needed
LABELS_PATH = "class_labels.json"
IMG_SIZE    = 224

@st.cache_resource
def load_model_and_labels():
    """Load model and label map once, cache for all sessions."""
    if not os.path.exists(MODEL_PATH):
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            label_map = json.load(f)
        # JSON keys are strings; convert to int
        label_map = {int(k): v for k, v in label_map.items()}
    else:
        # Fallback order if labels file missing
        label_map = {i: name for i, name in enumerate(DISEASE_INFO.keys())}

    return model, label_map

@st.cache_data
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for model inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)

def predict(model, img_array: np.ndarray, label_map: dict):
    """Run inference and return sorted results."""
    probs = model.predict(img_array, verbose=0)[0]
    results = [
        {"class": label_map[i], "confidence": float(probs[i])}
        for i in range(len(probs))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results

# ── RENDER CONFIDENCE BARS ────────────────────────────────────────────────
def render_confidence_bars(results):
    st.markdown("**Confidence breakdown:**")
    for r in results:
        name = r["class"]
        conf = r["confidence"]
        emoji = DISEASE_INFO.get(name, {}).get("emoji", "🔵")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<span class='confidence-bar-label'>{emoji} {name}</span>", unsafe_allow_html=True)
            st.progress(conf)
        with col2:
            st.markdown(f"<br><b>{conf*100:.1f}%</b>", unsafe_allow_html=True)

# ── MAIN APP ──────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("<div class='main-title'>🩺 Skin Doctor AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload a photo of the affected skin area for an AI-assisted assessment</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class='disclaimer'>
    ⚠️ <strong>Medical disclaimer:</strong> This tool is for educational purposes only.
    It is NOT a substitute for professional medical advice. Always consult a qualified
    dermatologist for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, label_map = load_model_and_labels()

    if model is None:
        st.error(f"❌ Model file `{MODEL_PATH}` not found. Please train and export your model first.")
        st.info("Run `python train_model.py` in the `model/` folder to train the model.")
        st.stop()

    # Upload section
    st.subheader("📷 Upload Skin Image")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a clear, well-lit photo of the affected area",
            type=["jpg", "jpeg", "png"],
            help="Best results with close-up, well-lit photos in natural light",
        )
    with col2:
        st.markdown("""
        **Tips for best results:**
        - Good natural lighting
        - Close-up shot (15–30 cm)
        - Avoid blurry photos
        - Single affected area
        """)

    if uploaded_file is None:
        # Show supported conditions
        st.subheader("🔍 Detectable Conditions")
        cols = st.columns(len(DISEASE_INFO))
        for col, (name, info) in zip(cols, DISEASE_INFO.items()):
            with col:
                st.markdown(f"**{info['emoji']} {name.replace('_', ' ')}**")
                st.caption(info['description'][:80] + "…")
        return

    # ── INFERENCE ────────────────────────────────────────────────────────
    image_bytes = uploaded_file.read()
    img_array   = preprocess_image(image_bytes)

    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.subheader("Uploaded image")
        st.image(image_bytes, use_column_width=True)

    with col_result:
        st.subheader("AI Assessment")
        with st.spinner("Analyzing..."):
            results = predict(model, img_array, label_map)

        top = results[0]
        top_conf = top["confidence"]
        top_class = top["class"]
        top_emoji = DISEASE_INFO.get(top_class, {}).get("emoji", "🔵")

        if top_conf >= 0.65:
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-title'>{top_emoji} {top_class.replace("_", " ")}</div>
                <div style='font-size:1.1rem; color:#1e8449;'>{top_conf*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)
        elif top_conf >= 0.4:
            st.warning(f"Possible **{top_class.replace('_', ' ')}** ({top_conf*100:.1f}%) — low confidence. Consider consulting a doctor.")
        else:
            st.markdown("""
            <div class='low-confidence'>
                ⚠️ Model is uncertain about this image. Please upload a clearer photo
                or consult a dermatologist.
            </div>
            """, unsafe_allow_html=True)

        render_confidence_bars(results)

    # ── DETAILED INFO ────────────────────────────────────────────────────
    if top_conf >= 0.4:
        info = DISEASE_INFO.get(top_class, {})
        st.subheader(f"ℹ️ About {top_class.replace('_', ' ')}")

        tab1, tab2, tab3 = st.tabs(["Description", "Treatment", "When to See a Doctor"])

        with tab1:
            st.markdown(f"**What it is:** {info.get('description', 'N/A')}")
            st.markdown(f"**Common causes:** {info.get('causes', 'N/A')}")

        with tab2:
            st.markdown(f"**Recommended treatment:** {info.get('treatment', 'N/A')}")
            st.markdown("""
            <div class='info-section'>
            💊 Always consult a pharmacist or doctor before starting any medication.
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown(f"{info.get('when_to_see_doctor', 'Please consult a dermatologist.')}")
            st.markdown("""
            <div class='info-section'>
            🏥 In Pakistan, you can find dermatologists at PIMS, Shifa International,
            Services Hospital, or book via oladoc.com / marham.pk
            </div>
            """, unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Model: MobileNetV2 fine-tuned on 625 skin images (5 classes). "
        "For educational use only. Not a medical device."
    )

if __name__ == "__main__":
    main()
