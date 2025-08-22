import streamlit as st
import numpy as np
import pickle

# Load the saved model
load_model = pickle.load(open('svm.sav', 'rb'))

def dia_pred(input_data):
    input_data_asarray = np.asarray(input_data)
    input_data_reshaped = input_data_asarray.reshape(1, -1)
    prediction = load_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'Not a Diabetic'
    else:
        return 'Is a Diabetic'

def main():
    st.title('Diabetes Prediction Page')

    # User input fields
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')

    diagnosis = ''

    if st.button('Diabetes Prediction Result'):
        diagnosis = dia_pred([Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin, BMI, DiabetesPedigreeFunction, Age])

        st.success(diagnosis)

if __name__ == '__main__':
    main()
