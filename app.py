import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

st.title('SMS Spam Detector')
st.write('This is a Machine Learning application to classify sms as well as mails as spam or ham (not spam).')

user_input = st.text_area('Enter your sms here:', height = 150)

if st.button('Detect', use_container_width=True):
    if user_input:
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result[0] == 0:
            st.write("Not Spam (ham)")
        else:
            st.write("Spam")

    else:
        st.write("Please write something to Detect")