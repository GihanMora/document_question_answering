import streamlit as st
import pandas as pd
import sqlite3
from sqlite3 import Connection
import openai
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import re
from dateutil.parser import parse
import traceback
from langchain.document_loaders import DirectoryLoader
import PyPDF2
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def read_and_textify(file):
#     pdfFileObj = open('/content/drive/MyDrive/ChatGPT/Resources/Data/LTU/24-Feb-2023-La-Trobe-University-UNGC-CoE-Report-FINAL.pdf', 'rb')
    pdfReader = PyPDF2.PdfReader (file)
    text_list = []
#     print("Page Number:", len(pdfReader.pages))
    for i in range(len(pdfReader.pages)):

      pageObj = pdfReader.pages[i]

      text = pageObj.extract_text()
      pageObj.clear()
#       print(text)

      # save to a text file for later use
      # copy the path where the script and pdf is placed
#       file1=open(r""+str(i)+"_convertedtext.txt","wb")
#       file1.writelines(text)
      text_list.append(text)



      # closing the text file object
#       file1.close()


    # closing the pdf file object
    file.close()
#     st.write(text_list)
    return text_list

def split_docs(documents,chunk_size=3000,chunk_overlap=100):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#   docs = text_splitter.split_documents(documents)
  docs = text_splitter.create_documents(documents)
  return docs

def get_similiar_docs(query,k=2,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs


def get_answer(query):
  similar_docs = get_similiar_docs(query)
  
  answer =  chain.run(input_documents=similar_docs, question=query)
  print('Answer >>>>>>>>>>')
  print(answer)
  st.write(answer)
  print("Relevant Documents >>>>>>>>>>")
  for d in similar_docs:
    print(d.metadata)
  # return  answer


# wide layout
st.set_page_config(layout="centered", page_title="Cooee + ChatGPT")

st.header("Document Question Answering")
st.write("---")
uploaded_file = st.file_uploader("Upload a documents", type=["txt","pdf"])
st.write("---")
if uploaded_file is None:
    st.info(f"""Upload a .pdf file to analyse""")


elif uploaded_file:
    #get text from documents
    documents = read_and_textify(uploaded_file)
    #text chunking
    docs = split_docs(documents)
    st.write(str(len(docs)) + " documents are loaded..")
    
    #extract embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
    
    
    pinecone.init(
    api_key="e32d4136-e020-4410-bfef-62031f37461d",  # find at app.pinecone.io
    environment="us-west1-gcp-free"  # next to api key in console
    )

    index_name = "document-qa"

    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    
    # model_name = "text-davinci-003"
    model_name = "gpt-3.5-turbo"
    # model_name = "gpt-4"
    llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"])
    
    chain = load_qa_chain(llm, chain_type="stuff")

    st.header("Ask your data")
    user_q = st.text_area("Enter your question here")
    if st.button("Get Response"):
            try:
                # create gpt prompt
                
                result = get_answer(user_q)
                st.subheader('Your response: {}'.format(result))
                

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
    
    
   
 
