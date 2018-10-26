# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:04:47 2018

@author: Supriya
"""

import numpy as np
import h5py
f1 = open('classes.txt','r')
exec(f1.read())
filename = 'GOLD_XYZ_OSC.0001_1024.hdf5'
f = h5py.File(filename, 'r')
#print(list(f.keys()))
dat = f['X']
det = f['Y'] #-- mods
dit = f['Z'] #-- snrs
#print (dat.shape, det.shape, dit.shape) #(2555904, 1024, 2) (2555904, 24) (2555904, 1)
nonzeroind = np.nonzero(det) 
#print (nonzeroind)
nonzero_row = nonzeroind[0]
nonzero_col = nonzeroind[1]
mods=[]
for row, col in zip(nonzero_row, nonzero_col):
    mods.append(classes[col])

snr = list(i[0] for i in dit)
ky=tuple(zip(mods,snr))
mod_data = {key: np.array(dat[i] for i in range(0, len(ky)) ) for key in ky}
