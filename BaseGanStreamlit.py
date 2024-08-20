import streamlit as st
import joblib
import numpy as np
from tensorflow import keras
#predict new image using the generator and save it as newgen
import matplotlib.pyplot as plt

# Load the generator model
gen = keras.models.load_model('gan_images_mnist/GAN_gen.h5  ')

# Generate a new image
generate=st.button('Generate New Image')
if generate:
    newgen = gen.predict(np.random.randn(1, 100))
    newgen = newgen.reshape(28, 28)

    # Display the new image using Streamlit
    
    plt.imshow(newgen, cmap='gray')
    plt.axis('off')
    st.pyplot(plt)
    generate=False