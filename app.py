import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('deploy_cnn.best.hdf5')
  return model

def import_and_predict(image_data, model):
        size = (150,150)  
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image, dtype = 'float32')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = img / 255
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        #img_reshape = img[np.newaxis,...]
        prediction = model.predict(img)
        return prediction

columns = ['mountain', 'street', 'glacier', 'building', 'sea', 'forest']
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Intel Image Classification
         """
         )

file = st.file_uploader("Please upload a 150x150 image file (PNG or JPG only)", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
if file is None:
    st.text("Please upload an image file")
else:
    size = (150,150)  
    image = Image.open(file)
    image = ImageOps.fit(image, size)
    st.image(image, width = image.size[0]*2)
    prediction = import_and_predict(image, model)
    #prediction = model.predict(image)
    score = tf.nn.softmax(prediction[0])
    #st.write(prediction)
    #st.write(score)
    string = "This image most likely a {} with a {:.2f} percent confidence.".format(columns[np.argmax(score)], 100 * np.max(score))
    st.success(string)
