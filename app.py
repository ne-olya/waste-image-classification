from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from ecosort.inference import load_checkpoint, predict

st.set_page_config(page_title="EcoSort", layout="wide")
st.title("EcoSort")
checkpoint_path = st.sidebar.text_input("Checkpoint", "checkpoints/best.pt")
uploaded = st.file_uploader("Фотография отхода", type=["jpg", "jpeg", "png", "webp"])
if uploaded and Path(checkpoint_path).exists():
    image = Image.open(uploaded).convert("RGB")
    model, checkpoint = load_checkpoint(checkpoint_path)
    result = predict(model, image, checkpoint["classes"], checkpoint["image_size"])
    left, right = st.columns(2)
    left.image(image, caption="Исходное изображение", use_container_width=True)
    right.image(result.heatmap, caption="Grad-CAM", use_container_width=True)
    st.metric("Класс", result.label, f"уверенность {result.confidence:.1%}")
    st.dataframe(
        pd.DataFrame(
            {"Класс": result.probabilities.keys(), "Вероятность": result.probabilities.values()}
        ).sort_values("Вероятность", ascending=False),
        hide_index=True,
        use_container_width=True,
    )
    st.info(result.recommendation)
elif uploaded:
    st.error(f"Checkpoint не найден: {checkpoint_path}")
