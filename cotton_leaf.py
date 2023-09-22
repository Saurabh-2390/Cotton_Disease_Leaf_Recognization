from tensorflow.keras.models import load_model
import streamlit as st
from fastapi import FastAPI
import numpy as np
from keras.utils.image_utils import load_img
from keras.utils.image_utils import img_to_array


model = load_model("C:\\Users\\Asus\\cotton_leaf.hdf5")
from keras.preprocessing.image import ImageDataGenerator


classes = ["bacterial_blight","curl_virus" ,"fussarium_wilt","healthy"]

app3_app = FastAPI()

def cotton_leaf_UI():
    st.title("Predict Type of cotton_leaf_Disease")
    uploaded_file = st.file_uploader("Choose a image file")
    ok = st.button("Predict")
    try:
        if ok == True:  # if user pressed ok button then True passed
            im1 = load_img(uploaded_file, target_size=(224, 224))
            x = img_to_array(im1)
            x = x * (1. / 255)
            x = np.expand_dims(x, axis=0)
            st.image(im1)
            result = classes[np.argmax(model.predict(x))]
            print(result)
           # classindx = predict_image(im1)
            if result == "bacterial_blight":
                st.error("cotton leaf disease  is bacterial_blight")
            elif result == "curl_virus":
                st.error("cotton leaf disease is curl_virus")
            elif result == "fussarium_wilt":
                st.error("cotton leaf disease is fussarium_wilt")
            elif result == "healthy":
                st.success("healthy leaf")
    except Exception as e: # all error
            st.info(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app3_app, host="0.0.0.0", port=8000,log_level='info')