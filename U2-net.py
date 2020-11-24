import tensorflow as tf
import os
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from dataLoader import getData

def loss(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)
    loss0 = bce(y_true, y_pred[0])
    loss1 = bce(y_true, y_pred[1])
    loss2 = bce(y_true, y_pred[2])
    loss3 = bce(y_true, y_pred[3])
    loss4 = bce(y_true, y_pred[4])
    loss5 = bce(y_true, y_pred[5])
    loss6 = bce(y_true, y_pred[6])
    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

def REBNCONV(x, out_ch=3, dirate=1):
    #x = ZeroPadding2D((1*dirate,1*dirate))(x)
    x = Conv2D(out_ch, 3, padding='same', dilation_rate = 1*dirate)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x

def _upsample_like(src, tar):
    h = int(tar.shape[1]/src.shape[1])
    w = int(tar.shape[2]/src.shape[2])
    src = UpSampling2D((h,w), interpolation='bilinear')(src)
    return src

def RSU7(x, mid_ch=12, out_ch=3):
    
    x0 = REBNCONV(x, out_ch, 1)
    
    x1 = REBNCONV(x0, mid_ch, 1)
    x = MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x3)

    x4 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x4)

    x5 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x5)

    x6 = REBNCONV(x, mid_ch, 1)

    x = REBNCONV(x6, mid_ch, 2)

    x = REBNCONV(tf.concat([x,x6],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x5)

    x = REBNCONV(tf.concat([x,x5],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x4)

    x = REBNCONV(tf.concat([x,x4],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x3)

    x = REBNCONV(tf.concat([x,x3],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x2)

    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x1)

    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def RSU6(x, mid_ch=12, out_ch=3):
    
    x0 = REBNCONV(x, out_ch, 1)
    
    x1 = REBNCONV(x0, mid_ch, 1)
    x = MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x3)

    x4 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x4)

    x5 = REBNCONV(x, mid_ch, 1)

    x = REBNCONV(x, mid_ch, 2)

    x = REBNCONV(tf.concat([x,x5],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x4)

    x = REBNCONV(tf.concat([x,x4],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x3)

    x = REBNCONV(tf.concat([x,x3],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x2)

    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x1)

    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def RSU5(x, mid_ch=12, out_ch=3):
    
    x0 = REBNCONV(x, out_ch, 1)
    
    x1 = REBNCONV(x0, mid_ch, 1)
    x = MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x3)

    x4 = REBNCONV(x, mid_ch, 1)

    x = REBNCONV(x, mid_ch, 2)

    x = REBNCONV(tf.concat([x,x4],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x3)

    x = REBNCONV(tf.concat([x,x3],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x2)

    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x1)

    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def RSU4(x, mid_ch=12, out_ch=3):
    
    x0 = REBNCONV(x, out_ch, 1)
    
    x1 = REBNCONV(x0, mid_ch, 1)
    x = MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)

    x = REBNCONV(x, mid_ch, 2)

    x = REBNCONV(tf.concat([x,x3],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x2)

    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x1)

    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def RSU4F(x, mid_ch=12, out_ch=3):

    x0 = REBNCONV(x, out_ch, 1)
    
    x1 = REBNCONV(x0, mid_ch, 1)
    x2 = REBNCONV(x1, mid_ch, 2)
    x3 = REBNCONV(x2, mid_ch, 4)
    
    x4 = REBNCONV(x3, mid_ch, 8)
    
    x = REBNCONV(tf.concat([x4,x3],axis=-1), mid_ch, 4)
    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 2)
    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def U2NET(x, out_ch=1):
    
    x1 = RSU7(x, 32, 64)
    x = MaxPool2D(2, 2)(x1)

    x2 = RSU6(x, 32, 128)
    x = MaxPool2D(2, 2)(x2)

    x3 = RSU5(x, 64, 256)
    x = MaxPool2D(2, 2)(x3)

    x4 = RSU4(x, 128, 512)
    x = MaxPool2D(2, 2)(x4)

    x5 = RSU4F(x, 256, 512)
    x = MaxPool2D(2, 2)(x5)

    x6 = RSU4F(x, 256, 512)
    x = _upsample_like(x6,x5)

    #-----------------decoder--------------------#

    x5 = RSU4F(tf.concat([x,x5],axis=-1),256, 512)
    x = _upsample_like(x5,x4)

    x4 = RSU4(tf.concat([x,x4],axis=-1),128, 256)
    x = _upsample_like(x4,x3)

    x3 = RSU5(tf.concat([x,x3],axis=-1),64, 128)
    x = _upsample_like(x3,x2)

    x2 = RSU6(tf.concat([x,x2],axis=-1),32, 64)
    x = _upsample_like(x2,x1)
    
    x1 = RSU7(tf.concat([x,x1],axis=-1),16, 64)

    #Side output
    x = ZeroPadding2D((1,1))(x1) 
    d1 = Conv2D(out_ch, 3)(x)
    d1 = Activation('sigmoid')(d1)

    x = ZeroPadding2D((1,1))(x2) 
    x = Conv2D(out_ch, 3)(x)
    d2 = _upsample_like(x,d1)
    d2 = Activation('sigmoid')(d2)
    
    x = ZeroPadding2D((1,1))(x3) 
    x = Conv2D(out_ch, 3)(x)
    d3 = _upsample_like(x,d1)
    d3 = Activation('sigmoid')(d3)
    
    x = ZeroPadding2D((1,1))(x4) 
    x = Conv2D(out_ch, 3)(x)
    d4 = _upsample_like(x,d1)
    d4 = Activation('sigmoid')(d4)
    
    x = ZeroPadding2D((1,1))(x5) 
    x = Conv2D(out_ch, 3)(x)
    d5 = _upsample_like(x,d1)
    d5 = Activation('sigmoid')(d5)
    
    x = ZeroPadding2D((1,1))(x6) 
    x = Conv2D(out_ch, 3)(x)
    d6 = _upsample_like(x,d1)
    d6 = Activation('sigmoid')(d6)

    d0 = Conv2D(out_ch, 1)(tf.concat([d1,d2,d3,d4,d5,d6],axis=-1))
    d0 = Activation('sigmoid')(d0)

    return tf.stack([d0,d1,d2,d3,d4,d5,d6])

def U2NETP(x, out_ch=1):
    
    x1 = RSU7(x, 16, 64)
    x = MaxPool2D(2, 2)(x1)

    x2 = RSU6(x, 16, 64)
    x = MaxPool2D(2, 2)(x2)

    x3 = RSU5(x, 16, 64)
    x = MaxPool2D(2, 2)(x3)

    x4 = RSU4(x, 16, 64)
    x = MaxPool2D(2, 2)(x4)

    x5 = RSU4(x, 16, 64)
    x = MaxPool2D(2, 2)(x5)

    x6 = RSU4F(x, 16, 64)
    x = _upsample_like(x6,x5)

    #---------------decoder--------------------
    x5 = RSU4F(tf.concat([x,x5],axis=-1),16, 64)
    x = _upsample_like(x5,x4)

    x4 = RSU4(tf.concat([x,x4],axis=-1),16, 64)
    x = _upsample_like(x4,x3)

    x3 = RSU5(tf.concat([x,x3],axis=-1),16, 64)
    x = _upsample_like(x3,x2)

    x2 = RSU6(tf.concat([x,x2],axis=-1),16, 64)
    x = _upsample_like(x2,x1)
    
    x1 = RSU7(tf.concat([x,x1],axis=-1),16, 64)

    x = ZeroPadding2D((1,1))(x1) 
    d1 = Conv2D(out_ch, 3)(x)
    d1 = Activation('sigmoid')(d1)

    x = ZeroPadding2D((1,1))(x2) 
    x = Conv2D(out_ch, 3)(x)
    d2 = _upsample_like(x,d1)
    d2 = Activation('sigmoid')(d2)
    
    x = ZeroPadding2D((1,1))(x3) 
    x = Conv2D(out_ch, 3)(x)
    d3 = _upsample_like(x,d1)
    d3 = Activation('sigmoid')(d3)
    
    x = ZeroPadding2D((1,1))(x4) 
    x = Conv2D(out_ch, 3)(x)
    d4 = _upsample_like(x,d1)
    d4 = Activation('sigmoid')(d4)
    
    x = ZeroPadding2D((1,1))(x5) 
    x = Conv2D(out_ch, 3)(x)
    d5 = _upsample_like(x,d1)
    d5 = Activation('sigmoid')(d5)
    
    x = ZeroPadding2D((1,1))(x6) 
    x = Conv2D(out_ch, 3)(x)
    d6 = _upsample_like(x,d1)
    d6 = Activation('sigmoid')(d6)

    d0 = Conv2D(out_ch, 1)(tf.concat([d1,d2,d3,d4,d5,d6],axis=-1))
    d0 = Activation('sigmoid')(d0)

    return tf.stack([d0,d1,d2,d3,d4,d5,d6]) 

net_input = Input(shape=(256,256,3)) 

model_output = U2NET(net_input)

model = Model(inputs = net_input, outputs = model_output)

opt = tf.keras.optimizers.Adam(lr = 1e-3)

bce = BinaryCrossentropy()

model.compile(optimizer = opt, loss = loss, metrics = None)

redu = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')

early = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=15, verbose=0, mode='auto')

image_path = os.path.join(os.getcwd(),'images')
gt_path = os.path.join(os.getcwd(), 'ground_truth_mask')

images, labels = getData(image_path, gt_path)

model.fit(images, labels, validation_split=0.2, epochs=10000, batch_size=16, callbacks=[redu,early], verbose=1, shuffle = True)
model.save('model.h5')
