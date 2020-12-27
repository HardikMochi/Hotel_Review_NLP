import pandas as pd
import numpy as np
import streamlit as st
import pickle

from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

word_index_dict = pickle.load(open('data/word_index_dict.pkl','rb'))

neural_net_model = load_model('data/Neural_Network.h5')

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

def add_sum_suffix(text):
    
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        word = word + '_sum'
        new_text += word + ' '
        
    return new_text

def index_review_words(text):
    review_word_list = []
    for word in text.lower().split():
        if word in word_index_dict.keys():
            review_word_list.append(word_index_dict[word])
        else:
            review_word_list.append(word_index_dict['<UNK>'])

    return review_word_list

def text_cleanup(text):
    print(text)
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        new_text += word + ' '
    print(new_text)    
    return new_text

def main():
    print("jjrj")
    st.title("Hotel Review Classifier Application")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Hotel Review Classifier Application </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    review_summary_text = st.text_input('Enter Your Review Summary Here')
    review_text = st.text_area('Enter Your Review Here')   
    print("wdjkafw")

    if st.button('predict'):
        print("djhafjk")
        result_review_sum = review_summary_text.title()
    
        result_review = review_text.title()
        review_summary_text = add_sum_suffix(review_summary_text)

        review_text = text_cleanup(review_text)

        review_text = index_review_words(review_text)

        review_summary_text = index_review_words(review_summary_text)
        
        all_review_text = review_text + review_summary_text

        all_review_text = sequence.pad_sequences([all_review_text],value=word_index_dict['<PAD>'],padding='post',maxlen=250)
        prediction = neural_net_model.predict(all_review_text)
    
        prediction = np.argmax(prediction)
        print("jwerhjkh")
        st.success(prediction+1)

if __name__  == "__main__":
    main()
