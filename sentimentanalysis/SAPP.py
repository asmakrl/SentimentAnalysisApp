import streamlit as st
from textblob import TextBlob

st.title("Sentiment Analysis App")
st.subheader("NLP")
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

                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                if sentiment.polarity > 0:
                    st.markdown("Sentiment: Positive :smiley: ")
                else:
                    st.markdown("Sentiment: Negative :angry: ")

            with col2:
                st.info("Word Sentiment")
                
 
else:
    st.subheader('About')