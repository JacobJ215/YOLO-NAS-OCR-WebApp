import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile


def main():
    st.title("License Plate Detection and OCR with YOLO-NAS")
    app_mode = st.sidebar.selectbox(
            'Choose Detection Mode', ['Run on Image', 'Run on Vide']
    )
    if app_mode == 'Run on Image':
        confidence = st.sidebar.slider(
            'Confidece', min_value=0.0, max_value=1.0
        )

if __name__ == '__main__':
    main()