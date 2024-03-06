import streamlit as st
import requests
import cv2
import numpy as np
import base64

st.set_page_config(layout="wide")

# Streamlit UI
st.title("Object Detection With FastAPI - Ha Khai Hoan")

url = "http://localhost:8000/detect"

col1_row1, col2_row1 = st.columns(2)

col1_row2,col2_row2, col3_row2 = st.columns((4, 1, 4))

with col1_row1:
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    with col1_row2:
        if uploaded_image is not None:
            st.text("Uploaded Image")
            st.image(uploaded_image, use_column_width=True)
            files = {"file": uploaded_image}
            response = requests.post(url,files=files)

with col2_row2:
    if st.button('Detect object'):
        with col3_row2:
            if response.status_code == 200:
                result = response.json()
                image_byte = result["image_detected"]          
                image_base64 = np.fromstring(base64.b64decode(image_byte), dtype=np.uint8)
                image = cv2.imdecode(image_base64, cv2.IMREAD_ANYCOLOR)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                st.text("Detected Image")
                st.image(image, use_column_width=True)
            else:
                st.error("Failed to detect the image. Please try again.")


