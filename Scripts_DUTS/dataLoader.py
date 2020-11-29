import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gc

def getData(train_path, gt_path, epoch, batch_size):
    X = []
    Y = []

    train_files = sorted(os.listdir(train_path))[epoch*batch_size:(epoch+1)*batch_size]
    gt_files = sorted(os.listdir(gt_path))[epoch*batch_size:(epoch+1)*batch_size]

    for i in range(len(train_files)):
        img = load_img(os.path.join(train_path, train_files[i]))
        img = img.resize((256,256))
        img = img_to_array(img)
        X.append(img)

        img = load_img(os.path.join(gt_path, gt_files[i]), color_mode='grayscale')
        img = img.resize((256,256))
        img = img_to_array(img)
        shape = img.shape
        img /= 255.0
        img = img.reshape(-1)
        idx = np.where(np.logical_and(img > 0.25, img < 0.8))[0] 
        if len(idx) > 0:
          img[idx] = -1
        img = img.reshape(shape)
        img = np.floor(img)
        Y.append(img)

    X = np.asarray(X)
    Y = np.asarray(Y)

    del train_files,gt_files, img
    gc.collect()
    return X, Y