# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:16:35 2018

@author: homepc
"""

# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
%matplotlib inline
import os,random
os.environ["KERAS_BACKEND"] = "theano"
# os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"]  = "floatX=float32,device=cpu,nvcc.flags=-D_FORCE_INLINES"
import numpy as np
import theano as th
import theano.tensor as T
from keras import backend as kBack
kBack.set_image_dim_ordering('th')
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
from collections import defaultdict
def create_model():
    dr = 0.5 # dropout rate (%)
    model = models.Sequential()
    model.add(Reshape([1]+in_shp, input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(80, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense( len(classes), init='he_normal', name="dense2" ))
    model.add(Activation('softmax'))
    #model.add(Reshape([len(classes)]))
    #model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    #model.summary()
    return model

Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'),encoding='bytes')

#Calculate signal to noise ratios (SNRS) and modulations (mods)

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

idx= list(set(range(0,X.shape[0])))

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y = to_onehot(list(map(lambda x: mods.index(lbl[x][0]),idx)))

in_shp = list(X.shape[1:])
classes = mods

model = create_model()
model.load_weights('convmodrecnets_CNN2_0.5.wts.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

X_result = []
#acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], idx))
    test_X_i = X[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y[np.where(np.array(test_SNRs)==snr)]
    #q = (list(l) for l in (np.where(np.array(test_SNRs)==snr)))
    pred_snrs=[]
    for i in range(0,test_X_i.shape[0]):
        pred_snrs.append(snr)
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    pred_mods=[]
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        pred_mods.append(classes[k])
        
    ky=tuple(zip(pred_mods,pred_snrs)) 
    
    for i in range(0,test_X_i.shape[0]):
        X_result.append((ky[i],test_X_i[i]))
        #X_result.setdefault(ky[i]).append(test_X_i[i])
        
d=defaultdict(list)
for a,b in X_result:
    if a not in d: d[a] = []
    d[a].append((b))
Xrd = dict((k, np.array(v)) for k, v in d.items())

    