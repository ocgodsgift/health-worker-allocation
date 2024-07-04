import streamlit as st
from PIL import Image
import time

# List of image paths or URLs
image_paths = [
    'IMG_0531.JPG',
    'IMG_0498.JPG',
]

def slideshow():

    # Set up the current image index in session state
    if 'current_image' not in st.session_state:
        st.session_state.current_image = 0

    # Function to go to the next image
    def next_image():
        st.session_state.current_image += 1
        if st.session_state.current_image >= len(image_paths):
            st.session_state.current_image = 0

    # Function to go to the previous image
    def prev_image():
        st.session_state.current_image -= 1
        if st.session_state.current_image < 0:
            st.session_state.current_image = len(image_paths) - 1

    # Display the current image
    image = Image.open(image_paths[st.session_state.current_image])
    st.image(image, use_column_width=True)

    # Display navigation buttons
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.button("Previous", on_click=prev_image)
    with col3:
        st.button("Next", on_click=next_image)
