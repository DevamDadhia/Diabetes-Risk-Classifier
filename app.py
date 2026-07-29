import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
import os

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="PixelGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# LOAD CSS
# -------------------------------------------------------

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

local_css("style.css")

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "pixelguard_resnet18.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()

# -------------------------------------------------------
# IMAGE TRANSFORMS
# -------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

classes = [
    "AI Generated",
    "Real"
]

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

def predict(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.softmax(outputs,1)

        confidence, pred = torch.max(probs,1)

    return (
        classes[pred.item()],
        confidence.item()*100,
        probs.squeeze().numpy()*100
    )

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown("""
<div class='hero'>

<h1>🛡️ PixelGuard AI</h1>

<p>
Deep Learning powered AI Image Detection
</p>

</div>
""",unsafe_allow_html=True)

# -------------------------------------------------------
# TWO COLUMN LAYOUT
# -------------------------------------------------------

left,right = st.columns([1,1])

# =======================================================
# LEFT PANEL
# =======================================================

with left:

    st.markdown("""
    <div class='glass'>

    <h2>📤 Upload Image</h2>

    Upload a JPG, JPEG or PNG image.

    </div>
    """,unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

# =======================================================
# RIGHT PANEL
# =======================================================

with right:

    st.markdown("""
    <div class='glass'>

    <h2>⚡ Detection</h2>

    Click below to start AI analysis.

    </div>
    """,unsafe_allow_html=True)

    if uploaded:

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True
        ):

            progress = st.progress(0)

            status = st.empty()

            steps = [
                "Loading model...",
                "Extracting features...",
                "Running CNN...",
                "Comparing patterns...",
                "Generating prediction..."
            ]

            for i,step in enumerate(steps):

                status.info(step)

                progress.progress(
                    int((i+1)/len(steps)*100)
                )

                time.sleep(0.5)

            prediction,confidence,probs = predict(image)

                        # -------------------------------
            # RESULT CARD
            # -------------------------------

            status.success("Analysis Complete!")

            if prediction == "Real":
                color = "#22C55E"
                icon = "✅"

            elif prediction == "AI Generated":
                color = "#F59E0B"
                icon = "🤖"

            else:
                color = "#EF4444"
                icon = "⚠️"

            st.markdown(
                f"""
                <div class="prediction-card" style="
                    border-left:6px solid {color};
                ">
                    <h2>{icon} {prediction}</h2>
                    <h1>{confidence:.2f}%</h1>
                    <p>Prediction Confidence</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📊 Class Probabilities")

            labels = [
                "AI Generated",
                "Fake",
                "Real"
            ]

            for label, value in zip(labels, probs):

                st.write(f"**{label}**")

                st.progress(float(value) / 100)

                st.caption(f"{value:.2f}%")

            st.divider()

            # -------------------------------
            # IMAGE INFORMATION
            # -------------------------------

            width, height = image.size

            st.markdown("## 🖼 Image Information")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Width",
                    f"{width}px"
                )

                st.metric(
                    "Height",
                    f"{height}px"
                )

            with col2:

                st.metric(
                    "Format",
                    image.format if image.format else "Unknown"
                )

                uploaded.seek(0)

                size = len(uploaded.read()) / (1024 * 1024)

                st.metric(
                    "Size",
                    f"{size:.2f} MB"
                )

            st.divider()

            # -------------------------------
            # MODEL INFO
            # -------------------------------

            st.markdown("## 🤖 Model")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.info("**Architecture**\n\nResNet18")

            with c2:
                st.info("**Framework**\n\nPyTorch")

            with c3:
                st.info("**Classes**\n\n3")

            st.divider()

            # -------------------------------
            # DETECTION GUIDE
            # -------------------------------

            with st.expander("💡 How to interpret the results"):

                st.markdown("""

### ✅ Real

The uploaded image appears to be captured naturally and does not exhibit obvious AI-generated artefacts.

---

### 🤖 AI Generated

The model detected patterns commonly associated with images created using generative AI models.

---

### ⚠️ Fake

The uploaded image may have been manipulated or synthetically altered.

---

Higher confidence generally indicates that the model is more certain about its prediction.

""")

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------

with st.sidebar:

    st.title("🛡 PixelGuard AI")

    st.markdown("---")

    st.write("### Model")

    st.success("ResNet18")

    st.write("### Framework")

    st.success("PyTorch")

    st.write("### Version")

    st.success("1.0")

    st.markdown("---")

    st.caption("Made with ❤️ using Streamlit")

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------

st.markdown("""

<br><br>

<div class="footer">

<h3>🛡 PixelGuard AI</h3>

<p>
AI Image Detection using Deep Learning
</p>

<p>
Built with Streamlit • PyTorch • ResNet18
</p>

</div>

""", unsafe_allow_html=True)