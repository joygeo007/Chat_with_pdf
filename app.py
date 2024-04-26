
from transformers import pipeline
from pypdf import PdfReader # used to extract text from pdf
from langchain.text_splitter import CharacterTextSplitter # split text in smaller snippets
#import os # read API key from environment variables. Not required if you are specifying the key in notebook.
from openai import OpenAI # used to access openai api
import json # used to create a json to store snippets and embeddings
from numpy import dot # used to match user questions with snippets.
import openai
from app_utils import *


#path of the folder containing the pdf files
folder_path="."

#find and read all pdf files into a text file
load_pdf(folder_path)

#create embeddings and write to json file
create_embeddings()





import streamlit as st
st.title("Helper Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    
    response = answer_users_question(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
