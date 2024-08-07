import os
import base64
import streamlit as st
import streamlit.components.v1 as components

# Function to create the HTML for the carousel
def slideshow():
    def get_carousel(image_paths):
        carousel_template = """
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel">
        <ol class="carousel-indicators">
            {indicators}
        </ol>
        <div class="carousel-inner">
            {items}
        </div>
        <a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">
            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
            <span class="sr-only">Previous</span>
        </a>
        <a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">
            <span class="carousel-control-next-icon" aria-hidden="true"></span>
            <span class="sr-only">Next</span>
        </a>
        </div>
        <style>
        .carousel-item img {{
            max-height: auto; /* Adjust to fit your needs */
            width: auto;
            margin: auto;
        }}
        </style>
        """

        indicators = ''.join([f'<li data-target="#carouselExampleIndicators" data-slide-to="{i}" {"class=active" if i==0 else ""}></li>' for i in range(len(image_paths))])
        items = ''.join([f'<div class="carousel-item {"active" if i==0 else ""}"><img src="data:image/jpg;base64,{image_paths[i]}" class="d-block w-100" alt="..."></div>' for i in range(len(image_paths))])

        return carousel_template.format(indicators=indicators, items=items)

    # Function to load images from the folder
    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                with open(os.path.join(folder, filename), "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                    images.append(encoded_string)
        return images

    # Load images from the 'images' folder
    image_folder = "images"
    image_paths = load_images_from_folder(image_folder)

    # Insert the carousel HTML into the Streamlit app
    if image_paths:
        carousel_html = get_carousel(image_paths)
        components.html(carousel_html, height=1000, width=1000)
    else:
        st.write("No images found in the folder.")
