import streamlit as st
import tensorflow as tf
import streamlit as st
from google_drive_downloader import GoogleDriveDownloader as gdd



@st.cache(allow_output_mutation=True)
def load_model():
  gdd.download_file_from_google_drive(file_id='19I0Q5zASzFaSlWgkXO0N5ZqJ3fqfkwFC',
                                    dest_path='./data/deploy_cnn.best.hdf5')
  model=tf.keras.models.load_model('./data/deploy_cnn.best.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Flower Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(prediction)
    st.write(score)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(columns[np.argmax(score)], 100 * np.max(score))
)
