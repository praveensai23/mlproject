# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:13:36 2025

@author: saipr
"""

import numpy as np
import pickle
import streamlit as st
load_model = pickle.load(open('classmodel.sav','rb'))

st.title('Diabetes prediction')

col1,col2,col3 = st.columns(3)
with col1:
    Pregnancies = st.text_iput('No.of Pregnancies')
with col2:
    Glucose = st.text_iput('Glucose Levels')
with col3:
    BloodPressure = st.text_iput('Blood Pressure Value')
with col1:
    SkinThickness = st.text_iput('Skin Thickness value')
with col2:
    Insulin = st.text_iput('Insulin value')
with col3:
    BMI = st.text_iput('BMI value')
with col1:
    DiabetesPedigreeFunction = st.text_iput('Diabetes Pedigree Function value')
with col2:
    Age = st.text_iput('Age of the person')
    
    dia_diag = ''
    
if st.button('Result'):
    dia_pred = load_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    if (dia_pred[0]==1):
       dia_diag = 'He is diabetic'
    else:
       dia_diag = 'He is Not a diabetic'
       
st.success(dia_diag)
       
      