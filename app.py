import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2
import glob
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
        return tf.keras.models.load_model('deploy_cnn.best.hdf5')
        
def import_and_predict(image_data, model):
        size = (150,150)  
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image, dtype = 'float32')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = img / 255
        prediction = model.predict(img)
        return prediction

def display_images():
  images = [Image.open(file) for file in glob.glob("display/*.jpg")]
  row_size = len(images)
  grid = st.columns(row_size)
  col = 0
  for image in images:
      with grid[col]:
          st.image(image)
      col = (col + 1) % row_size

columns = ['mountain', 'street', 'glacier', 'building', 'sea', 'forest']
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("Emerging Technologies 2 by Pagatpat, Paul Gabriel and Dalangan, Katherine May")
display_images()
st.write("""
         # Intel Image Classification
         \nA demonstration on a Predictive Convolutional Neural Network with a 66% accuracy that uses
         images of natural scenes from a Datahack challenge by Intel.
         """
         )

file = st.file_uploader("Upload images that either classify as an image of a mountain, street, glacier, building, sea, or a forest (PNG or JPG only)", type=["jpg", "png"])
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
    string = "This image most likely a {} with a {:.2f}% confidence.".format(columns[np.argmax(score)], 100 * np.max(score))
    st.success(string)
