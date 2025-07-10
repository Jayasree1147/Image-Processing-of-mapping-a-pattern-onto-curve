import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- PAGE CONFIG & STYLE ---

st.set_page_config(page_title="🧵 Pattern-to-Flag Mapper", layout="centered")

# Custom background and font styling
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .stApp {
        background-image: linear-gradient(135deg, #f0f4f8, #d9e2ec);
        padding: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.4rem;
        color: #1f4e79;
        font-weight: 700;
    }
    .subhead {
        font-size: 1.1rem;
        color: #444;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stFileUploader > label {
        font-weight: 600;
        font-size: 1rem;
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stDownloadButton>button {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title">🏁 Pattern Mapping on a Folded Flag</div>', unsafe_allow_html=True)
st.markdown('<div class="subhead">Upload a white flag image (with folds) and a pattern to realistically map the design onto the fabric.</div>', unsafe_allow_html=True)

# --- Utility Functions ---

def apply_displacement_map(pattern, flag_gray, strength=0.3):
    h, w = pattern.shape[:2]
    grad_x = cv2.Sobel(flag_gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(flag_gray, cv2.CV_32F, 0, 1, ksize=5)

    dx = cv2.GaussianBlur(grad_x, (0, 0), 3)
    dy = cv2.GaussianBlur(grad_y, (0, 0), 3)

    dx = cv2.normalize(dx, None, -strength, strength, cv2.NORM_MINMAX)
    dy = cv2.normalize(dy, None, -strength, strength, cv2.NORM_MINMAX)

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.float32(map_x + dx * w)
    map_y = np.float32(map_y + dy * h)

    return cv2.remap(pattern, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def blend_images(base_img, overlay_img, alpha_mask):
    overlay = overlay_img.astype(float)
    base = base_img.astype(float)
    alpha = alpha_mask.astype(float) / 255.0
    if len(alpha.shape) == 2:
        alpha = cv2.merge([alpha, alpha, alpha])
    return cv2.convertScaleAbs(base * (1 - alpha) + overlay * alpha)

# --- File Uploads ---

flag_file = st.file_uploader("📤 Upload White Flag Image (with visible folds)", type=["jpg", "jpeg", "png"])
pattern_file = st.file_uploader("🖼️ Upload Pattern Image (rectangle, logo or design)", type=["jpg", "jpeg", "png"])

# --- Processing ---

if flag_file and pattern_file:
    flag_pil = Image.open(flag_file).convert("RGB")
    pattern_pil = Image.open(pattern_file).convert("RGB")

    flag = np.array(flag_pil)
    pattern = np.array(pattern_pil)

    pattern_resized = cv2.resize(pattern, (flag.shape[1], flag.shape[0]))
    flag_gray = cv2.cvtColor(flag, cv2.COLOR_RGB2GRAY)

    warped_pattern = apply_displacement_map(pattern_resized, flag_gray, strength=0.35)

    contrast = cv2.equalizeHist(flag_gray)
    alpha_mask = cv2.GaussianBlur(contrast, (0, 0), 7)
    alpha_mask = cv2.normalize(alpha_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    output = blend_images(flag, warped_pattern, alpha_mask)

    st.subheader("✅ Mapped Output")
    st.image(output, caption="Pattern Aligned with Flag Folds", use_container_width=True)

    # Download button
    result = Image.fromarray(output)
    buf = result.convert("RGB").tobytes("jpeg", "RGB")
    st.download_button("📥 Download Output", data=buf, file_name="Output.jpg", mime="image/jpeg")

else:
    st.warning("⚠️ Please upload both a white flag and a pattern image to continue.")
