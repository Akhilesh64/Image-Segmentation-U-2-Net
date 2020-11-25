import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from sklearn.preprocessing import binarize

img = Image.open(os.path.join(os.getcwd(),'images','0001.jpg')) #Example
img = img.resize((256,256))
img = np.array(img)
img = np.expand_dims(img,axis=0)

model = load_model('model.h5', compile=False)
preds = model.predict(img)
preds = np.squeeze(preds)

#For thresholding 
for i in range(len(preds)):
    shape = preds[i,:,:].shape
    frame = binarize(preds[1,:,:], threshold = 0.5)
    frame = np.reshape(frame,(shape[0], shape[1]))

#For saving all the frames
for i in range(len(preds)):
  img = np.expand_dims(np.array(preds[i,:,:]),axis=-1)
  img = array_to_img(img)
  img.save('img'+str(i)+'.png')
