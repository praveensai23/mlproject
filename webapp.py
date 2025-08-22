# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:13:36 2025

@author: saipr
"""

import numpy as np
import pickle
import streamlit as st
load_model = pickle.load(open('C:/Users/saipr/OneDrive/Attachments/svm.sav','rb'))
def dia_pred(input_data):
    input_data = (10,168,74,0,0,38,0.537,34)
    input_data_asarray = np.asarray(input_data)
    input_data_asarray_reshape = input_data_asarray.reshape(1,-1)
    pred = load_model.predict(input_data_asarray_reshape)
    if (pred==0):
      return 'not a diabetic'
    else:
      return 'Is a diabetic'
def main():
    st.title('Diabetes Prediction Page')
    Pregnancies = st.text_input('No.of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of person')
    diagnosis = ' '
    
    if st.button('Diabetes Prediction Result'):
       diagnosis = dia_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
       
    st.success(diagnosis)
       
if (__name__ == '__main__'):
   main()
    