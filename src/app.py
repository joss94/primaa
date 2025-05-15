import numpy as np
import streamlit as st

from wsi_parser import WSIParser
from PIL import Image

# Create the parser object
# Ideally, it would be better to call a backend via an API, but to keep it simple
# the model is created and handled directly from here
parser = WSIParser()

st.title('Primaa WSI nuclei masking')

# Get file from user selection
uploaded_file = st.file_uploader(label='Upload WSI data', type='ndpi')

if uploaded_file is not None:

    # Write bytes to temporary local file, since tiatoolbox cannot open bytes directly
    with open("tmp.ndpi", "wb") as binary_file:
        binary_file.write(uploaded_file.getvalue())

    # Process the file and show the resulting mask in an image
    if st.button('Run'):
        progress_bar = st.progress(0)
        mask_ph = st.empty()
        thumb_ph = st.empty()
        for progress, mask, thumb in parser.parse_slide_progress("tmp.ndpi"):
            progress_bar.progress(progress)
            thumb_ph.image(Image.fromarray(thumb).convert('RGB'))
            mask_ph.image(Image.fromarray(mask).convert('RGB'))