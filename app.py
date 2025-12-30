# app.py — FINAL STREAMLIT APP

import streamlit as st
import tempfile
import os
from PIL import Image
from predict import predict_tb

st.set_page_config(
    page_title="AI TB Screening Tool",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 AI-Powered Tuberculosis Screening")

st.markdown("""
This AI tool **screens chest X-ray images** to identify **potential TB risk**.

⚠️ **Disclaimer:**  
This is an **AI-assisted screening tool**, **not a diagnostic system**.  
Final diagnosis must be made by a qualified medical professional.
""")

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(uploaded_file.read())
        image_path = temp.name

    st.image(uploaded_file, caption="Uploaded Chest X-ray", use_container_width=True)

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Analyzing image..."):
            try:
                result = predict_tb(image_path)

                st.subheader(f"Screening Result: {result['risk']}")
                st.progress(result["probability"])
                st.caption(f"TB Probability: {result['probability']*100:.1f}%")
                st.caption(f"Confidence: {result['confidence']*100:.1f}%")
                st.info(result["advice"])

                if result["risk"] == "High TB Risk":
                    st.warning("⚠️ High risk detected. Immediate follow-up advised.")
                elif result["risk"] == "Moderate TB Risk":
                    st.info("ℹ️ Moderate risk. Clinical correlation suggested.")
                else:
                    st.success("✅ Low TB risk detected.")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    # cleanup
    if os.path.exists(image_path):
        os.unlink(image_path)

st.markdown("---")
st.caption("AI-assisted screening | Hackathon demo | Educational use only")
