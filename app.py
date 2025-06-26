import streamlit as st
from pathlib import Path
import google.generativeai as genai
import logging
import io
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.info("SDK imported successfully")

# Load your Gemini API key
from api_key import api_key

# Configure Gemini API
genai.configure(api_key=api_key)

# Generation and safety config
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# System prompt
system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with the following:

Your Responsibilities include:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal features.
2. Findings Report: Document all observed anomalies or signs of disease. Clearly articulate them.
3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including further testing or referrals.
4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:

1. Scope of Response: Only respond if the image pertains to human health issues.
2. Clarity of Image: In cases where the image quality impedes clear analysis, note that carefully.
3. Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."
4. Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis thoughtfully.

Please provide me an output response with these 4 headings:

1. Detailed Analysis  
2. Findings Report  
3. Recommendations and Next Steps  
4. Treatment Suggestions
"""

# Load model
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Streamlit page setup
st.set_page_config(page_title="Visionary Insights", page_icon="🧠", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
    }
    .stApp {
        max-width: 800px;
        margin: auto;
        padding: 2rem;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
    }
    h1, h3 {
        color: #0d3b66;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("🧠 Visionary Insights")
st.markdown("#### _AI-Powered Medical Imaging Assistant_")
st.markdown("Upload a medical image and receive detailed clinical insights in seconds. This tool is powered by Gemini and trained to assist in diagnostics. Please consult a physician before making any decisions.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your medical image", type=["jpg", "png", "jpeg"])

# Show uploaded image
if uploaded_file:
    st.image(uploaded_file, width=300, caption="✅ Uploaded Image Preview")

# Submit button
submit_button = st.button("🔍 Generate the Analysis")

# Main logic
if submit_button:
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()

        image_parts = [
            {"mime_type": "image/jpeg", "data": image_data},
        ]

        prompt_parts = [
            image_parts[0],
            system_prompt,
        ]

        with st.spinner("🧬 Running image diagnostics..."):
            response = model.generate_content(prompt_parts)

        if response:
            st.markdown("## 📝 Here is the analysis based on your image:")
            st.write(response.text)
        else:
            st.error("❌ No response was generated. Please try again.")
    else:
        st.warning("⚠️ Please upload a medical image before generating the analysis.")
