import streamlit as st
from textblob import TextBlob
import numpy as np
import tensorflow as tf
import pickle

@st.cache(allow_output_mutation=True)

def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


    
def predict(data):
    model=tf.keras.models.load_model('model2.h5')  
    return int(model.predict(data).round().item())

st.title("Sentiment Analysis App")
st.subheader("Natural Language Processing")
menu = ['Home','About']

choice = st.sidebar.selectbox("Menu", menu)
if choice == "Home":
    st.subheader("Home")
    with st.form(key = "nlpForm"):
        raw_text = st.text_area("Enter Text Here:")
        submit_button = st.form_submit_button(label = "Analyze")

        #layout
        col1, col2 = st.columns(2)

        if submit_button:
            with col1:

                st.info("My Model Results")
                tokenizer = load_tokenizer()
                seq = tokenizer.texts_to_sequences([raw_text])
                padded = tf.keras.preprocessing.sequence.pad_sequences(seq,maxlen=200)
                
                sentiment = predict(padded)
                st.write(sentiment)
                
                if sentiment == 1:
                    st.markdown("Sentiment: Positive :smiley: ")
                else:
                    st.markdown("Sentiment: Negative :angry: ")

            with col2:
                st.info("TextBlob Results")
                sentiment = TextBlob(raw_text).sentiment 
                st.write(sentiment)
            
                if sentiment.polarity>=0:
                    st.markdown("Sentiment: Positive :smiley: ")
                else:
                    st.markdown("Sentiment: Negative :angry: ")

else:
    st.subheader('About')
    st.info("This is a sentiment analysis app, you can find the jupiter notebook of the model with the code for the web app in my github repository via this link: https://github.com/asmakrl/SentimentAnalysisApp")