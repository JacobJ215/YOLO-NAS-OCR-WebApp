# Import necessary libraries
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
from license_plate_detection import *

# Define the main function
def main():
    # Set the title of the Streamlit app
    st.title("License Plate Detection and OCR with YOLO-NAS")

    # Create a selection box in the sidebar to choose detection mode (image or video)
    app_mode = st.sidebar.selectbox(
            'Choose Detection Mode', ['Run on Image', 'Run on Video']
    )

    # If the chosen mode is "Run on Image"
    if app_mode == 'Run on Image':
        # Create a slider to adjust confidence level
        confidence = st.sidebar.slider(
            'Confidence', min_value=0.0, max_value=1.0
        )

        # Create a file uploader for image files
        img_file = st.sidebar.file_uploader(
            "Upload an Image", type=["png", "jpg", "jpeg"]
        )

        # Default image for demonstration purposes
        demo = "./OCR-2/valid/images/195_jpg.rf.609d4d0dfd006181eb7d52a90c00bfd8.jpg"

        # Load the image from the file uploader or use the default demo image
        if img_file is not None:
            img = cv2.imdecode(np.fromstring(
                img_file.read(), np.uint8), 1
            )
            image = np.array(Image.open(img_file))
        else:
            img = cv2.imread(demo)
            image = np.array(Image.open(demo))

        # Display the original image in the sidebar
        st.sidebar.text("Original Image")
        st.sidebar.image(image)

        # Perform object detection and OCR on the image
        object_detection_ocr(img, confidence)

    # If the chosen mode is "Run on Video"
    elif app_mode == "Run on Video":
        # Create a slider to adjust confidence level
        confidence = st.sidebar.slider(
            'Confidence', min_value=0.0, max_value=1.0
        )

        # Create a file uploader for video files
        video_file = st.sidebar.file_uploader(
            "Upload a Video", type=["mp4", "avi", "mov", "asf"]
        )

        # Default video for demonstration purposes
        demo = "./Videos/demo.mp4"

        # Create a temporary file to store the uploaded video
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        # If no video is uploaded, display the demo video in the sidebar
        if not video_file:
            video_stream = cv2.VideoCapture(video_file)
            tffile.name = demo
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text("Input Video")
            st.sidebar.video(demo_bytes)
        else:
            # Write the uploaded video to the temporary file
            tffile.write(video_file.read())
            vid = open(tffile.name, 'rb')
            vid_bytes = vid.read()
            st.sidebar.text("Input Video")
            st.sidebar.video(vid_bytes)

        # Create an empty space to display video frames
        stframe = st.empty()

        # Create key performance indicators (KPIs) to display frame rate and license plate number
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**License Plate Number**")
            kpi2_text = st.markdown("0")

        # Perform object detection and OCR on the uploaded video
        object_detction__ocr_video(tffile.name, confidence, kpi1_text, kpi2_text, stframe)


# Execute the main function if this script is run directly
if __name__ == '__main__':
    main()
