
import streamlit as st
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import tempfile
import os
from PIL import Image

MODEL_PATH  = "music_mood_model.h5"
LABELS_PATH = "class_indices.json"
IMG_SIZE    = (224, 224)

MOOD_EMOJI = {
    "Happy":      "😄",
    "Sad":        "😢",
    "Romantic":   "💕",
    "Party":      "🎉",
    "Devotional": "🙏"
}
MOOD_COLOR = {
    "Happy":      "#FFD700",
    "Sad":        "#6495ED",
    "Romantic":   "#FF69B4",
    "Party":      "#FF4500",
    "Devotional": "#9370DB"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    with open(LABELS_PATH) as f:
        idx = json.load(f)
    return {v: k for k, v in idx.items()}

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, duration=30, sr=22050)
    S     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB  = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax  = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.axis("off")
    plt.tight_layout(pad=0)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = Image.open(tmp.name).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    os.unlink(tmp.name)
    return np.expand_dims(arr, axis=0), S_dB, sr

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Mood Classifier",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Music Mood Classifier")
st.markdown("### Powered by CNN + MobileNetV2 | Bollywood Song Mood Detection")
st.markdown("Upload a Bollywood song (`.wav` or `.mp3`) and the AI will predict its mood!")
st.divider()

model  = load_model()
labels = load_labels()

uploaded = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"],
    help="Supports .wav and .mp3 formats"
)

if uploaded:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**🎧 Uploaded Audio**")
        st.audio(uploaded)

    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(uploaded.name)[1], delete=False
    ) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing audio..."):
        try:
            img_arr, S_dB, sr = audio_to_spectrogram(tmp_path)

            with col2:
                st.markdown("**Mel Spectrogram**")
                fig, ax = plt.subplots(figsize=(4, 3))
                librosa.display.specshow(S_dB, sr=sr,
                    x_axis="time", y_axis="mel", ax=ax)
                plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
                ax.set_title("Mel Spectrogram", fontsize=9)
                st.pyplot(fig)
                plt.close(fig)

            preds   = model.predict(img_arr, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            mood    = labels[top_idx]
            conf    = preds[top_idx] * 100

            st.divider()
            st.markdown(f"""
            <div style="text-align:center; padding:20px;
                        background-color:{MOOD_COLOR.get(mood,"#eee")}22;
                        border-radius:12px; border:2px solid {MOOD_COLOR.get(mood,"#ccc")}">
                <h1>{MOOD_EMOJI.get(mood,"")} {mood}</h1>
                <h3>Confidence: {conf:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**All Mood Probabilities**")
            for i in range(len(preds)):
                lbl  = labels[i]
                prob = float(preds[i])
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(prob, text=f"{MOOD_EMOJI.get(lbl,'')} {lbl}")
                with col_b:
                    st.write(f"{prob*100:.1f}%")

        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            os.unlink(tmp_path)
