import streamlit as st
import tempfile
from predict import predict_tb

st.set_page_config(
    page_title="AI TB Screening Tool",
    layout="centered"
)

st.title("🫁 AI-Based Tuberculosis Screening Tool")

st.markdown("""
This tool uses **Artificial Intelligence** to **screen chest X-ray images**
and identify **potential TB risk**.

⚠️ *This is NOT a diagnostic system. Final diagnosis must be done by a medical professional.*
""")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(uploaded_file.read())
        image_path = temp.name

    risk, confidence = predict_tb(image_path)

    st.image(uploaded_file, caption="Uploaded Chest X-ray", use_column_width=True)

    st.subheader(f"Screening Result: {risk}")
    st.write(f"Model Confidence Score: **{confidence:.2f}**")

    if risk == "High TB Risk":
        st.warning("⚠️ High risk detected. Recommend confirmatory medical testing.")
    elif risk == "Moderate TB Risk":
        st.info("ℹ️ Moderate risk. Clinical correlation advised.")
    else:
        st.success("✅ Low risk detected.")

    st.markdown("---")
    st.caption("AI-assisted screening | For research and educational purposes only")
