import os
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input
import streamlit as st

# Initialize Streamlit app
st.title("X-Ray Analysis App")

# Load your models and categories
model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")
categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

# Define a function to predict
def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str

# Define a function to check pneumonia
def check_pneumonia(file):
    model = tf.keras.models.load_model('weights/our_model.h5')
    img = image.load_img(file, target_size=(224, 224))
    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)
    img_data = preprocess_input(image_arr)
    prediction = model.predict(img_data)

    if prediction[0][0] > prediction[0][1]:
        return 'Person is safe.'
    else:
        return 'Person is affected with Pneumonia.'

# Streamlit app
app_mode = st.sidebar.selectbox("Choose a mode:", ["Home", "Check Pneumonia", "Check Fracture"])

if app_mode == "Home":
    st.header("Welcome to the X-Ray Analysis App")
    st.write("Select a mode from the sidebar to get started.")

elif app_mode == "Check Pneumonia":
    st.header("Check for Pneumonia")
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded X-ray Image.", use_column_width=True)
        if st.button("Check"):
            result = check_pneumonia(uploaded_file)
            st.write(result)

elif app_mode == "Check Fracture":
    st.header("Check for Bone Fracture")
    uploaded_file = st.file_uploader("Upload an X-ray image of a body part", type=["png", "jpg", "jpeg"])
    body_part = st.selectbox("Select the body part:", ["Elbow", "Hand", "Shoulder"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded X-ray Image.", use_column_width=True)
        if st.button("Check"):
            result = predict(uploaded_file, body_part)
            st.write(f"Scanned X-Ray of {body_part} and it is observed to be {result}")

