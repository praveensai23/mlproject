# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:11:12 2025

@author: saipr
"""

import numpy as np
import pickle
load_model = pickle.load(open('C:/Users/saipr/OneDrive/Attachments/svm.sav','rb'))
input_data = (10,168,74,0,0,38,0.537,34)
input_data_asarray = np.asarray(input_data)
input_data_asarray_reshape = input_data_asarray.reshape(1,-1)
pred = load_model.predict(input_data_asarray_reshape)
if (pred==0):
  print('not a diabetic')
else:
  print('Is a diabetic')