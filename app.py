pip install streamlit

import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Feature scaling (optional if your model was trained with scaled features)
def scale_features(features):
    # Example scaling logic, replace with your actual scaling
    return (features - np.mean(features)) / np.std(features)

# Define the Streamlit app
def main():
    st.title('Titanic Survival Prediction')

    # Collect user input
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age = st.slider('Age', 0, 100, 25)
    fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=50.0)
    embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

    # Encode the categorical variables
    sex = 1 if sex == 'Male' else 0
    embarked_C = 1 if embarked == 'C' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    # Prepare features for prediction
    features = np.array([[pclass, sex, age, fare, embarked_C, embarked_Q, embarked_S]])

    # Prediction button
    if st.button('Predict'):
        prediction = model.predict(features)[0]
        prediction_prob = model.predict_proba(features)[0][1]  # Probability of survival

        if prediction == 1:
            st.success(f'This passenger would have survived with a probability of {prediction_prob:.2f}')
        else:
            st.error(f'This passenger would not have survived with a probability of {1 - prediction_prob:.2f}')

if __name__ == '__main__':
    main()
