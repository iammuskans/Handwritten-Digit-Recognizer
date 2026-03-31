import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
import time
import pandas as pd

# Load model
model = tf.keras.models.load_model("model.h5")

st.set_page_config(page_title="AI Digit Recognizer", layout="centered")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Glass UI
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.glass{
background: rgba(255,255,255,0.08);
padding:20px;
border-radius:15px;
backdrop-filter: blur(10px);
border:1px solid rgba(255,255,255,0.2);
}

</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Handwritten Digit Recognizer")

st.markdown('<div class="glass">Draw a digit between 0 and 9</div>', unsafe_allow_html=True)

# Clear canvas
if st.button("Clear Canvas"):
    st.rerun()

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:

    img = canvas_result.image_data

    img = Image.fromarray((img[:,:,0]).astype("uint8"))
    img = img.resize((28,28))

    img = np.array(img)/255.0
    img = img.reshape(1,28,28,1)

    with st.spinner("AI analyzing..."):
        time.sleep(1)

    prediction = model.predict(img)

    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))*100

    st.success(f"Prediction: {digit}")

    # Save to history
    st.session_state.history.append({
        "Digit": digit,
        "Confidence (%)": round(confidence,2)
    })

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "AI Confidence"},
        gauge={'axis': {'range':[0,100]}}
    ))

    st.plotly_chart(fig)

    # Top predictions
    st.subheader("Top Predictions")

    top3 = np.argsort(prediction[0])[-3:][::-1]

    for i in top3:
        prob = prediction[0][i]
        st.progress(float(prob))
        st.write(f"Digit {i} — {prob*100:.2f}%")

# Prediction history
if st.session_state.history:
    st.subheader("📊 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df)