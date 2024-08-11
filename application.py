import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
# from tensorflow.keras import models

# Function to preprocess uploaded image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # Normalize pixel values to range [0, 1]
    return img

# Load trained model
model = tf.keras.models.load_model('Model/BONE.h5')  # Replace 'your_model.h5' with the path to your saved model

# Streamlit app
def main():
    st.title('Fracture Detection App')

    

    
    st.write('Upload an image to check if it contains a fracture.')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', width = 250 )

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(np.expand_dims(processed_image, axis=0))[0][0]

        if prediction >= 0.5:
            st.write('Prediction: Fracture Detected') 
        else:
            st.write('Prediction: Non-Fractured')

    
if __name__ == '__main__':
    main()
