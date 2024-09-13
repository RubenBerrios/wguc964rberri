import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np


plant_names_Invasive = ['daisy','dandelion']
plant_names_Not_Invasive = ['roses','sunflowers','tulips']
plant_names_all = plant_names_Invasive + plant_names_Not_Invasive


model = load_model('plant_classify_model.keras')



#create tabs for main page and visualization
tab1, tab2, tab3 = st.tabs(["Prediction", "Visualization", "Help/Instructions"])

#tab for uploading images to predict
with tab1:
    st.title('Plant Classification CNN model')
    #model prediction method
    def classify_images(image_path):
        input_image = tf.keras.utils.load_img(
            image_path,
            color_mode='rgb', 
            target_size=(180,180),
            interpolation='nearest',
            keep_aspect_ratio=False)
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array,0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        checkIfPlantIsInvasive(plant_names_all[np.argmax(result)])
        outcome = 'This Plant type is: ' + plant_names_all[np.argmax(result)]
        
        
        return outcome

    #method to check if plant is invasive
    def checkIfPlantIsInvasive(plantName):
        #check if the plant name outputted by ML model is in the invasive list, otherwise it will be in not invasive list
        if plantName in plant_names_Invasive:
            st.error('The Plant is Invasive!', icon="🚨")
        else:
            st.success('The Plant is NOT invasive!', icon="✅")

    #upload file
    uploaded_file = st.file_uploader('Choose a file to upload', type=["jpeg"])
    if uploaded_file is not None:
        
        #display the image on web app
        st.image(uploaded_file, width=240)
        st.info(classify_images(uploaded_file))
        

#tab for displaying visualizations
with tab2:
    st.title("Visualizations")
    st.image('Visualization1.jpeg', width=500)
    st.image('Visualization2.jpeg', width=500)
    st.image('Visualization3.jpeg', width=500)


    




#tab for Instructions
with tab3:
    st.title("Licenses")
    


