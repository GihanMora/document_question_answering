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
    print("Page Number:", len(pdfReader.pages))
    for i in range(len(pdfReader.pages)):

      pageObj = pdfReader.pages[i]

      text = pageObj.extract_text()
      pageObj.clear()
      print(text)

      # save to a text file for later use
      # copy the path where the script and pdf is placed
#       file1=open(r""+str(i)+"_convertedtext.txt","wb")
#       file1.writelines(text)
      text_list.append(text)



      # closing the text file object
#       file1.close()


    # closing the pdf file object
    file.close()
    st.write(text_list)
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
    st.write(len(docs))
    
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
                st.write("asasasasasa")
                gpt_input = 'Write a sql lite query based on this question: {} The table name is my_table and the table has the following columns: {}. ' \
                            'Return only a sql query and nothing else'.format(user_q, cols)

                
                query = generate_gpt_reponse(gpt_input, max_tokens=200)

                
                
                query_clean = extract_code(query)
                result = run_query(conn, query_clean)

                with col1.expander("SQL query used for the question"):
                    col1.code(query_clean)

                # if result df has one row and one column
                if result.shape == (1, 1):

                    # get the value of the first row of the first column
                    val = result.iloc[0, 0]

                    # write one liner response
                    col1.subheader('Your response: {}'.format(val))

                else:
                    col1.subheader("Your result:")
                    col1.table(result)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                col1.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
    
    
    
    
    
    col2.header("Visualization")
    user_input = col2.text_area("Add features that the plot would have.")

    if col2.button("Create a visualization"):
        try:
            # create gpt prompt
            gpt_input = 'Write code in Python using Plotly to address the following request: {} ' \
                        'Use df that has the following columns: {}. Do not use animation_group argument and return only code with no import statements and the data has been already loaded in a df variable'.format(user_input, cols)

            with st.spinner('ChatGPT is working...'):
                gpt_response = generate_gpt_reponse(gpt_input, max_tokens=1500)

                extracted_code = extract_code(gpt_response)

                extracted_code = extracted_code.replace('fig.show()', 'col2.plotly_chart(fig)')

                with col2.expander("Plotly code used for the visualization"):
                    col2.code(extracted_code)

                # execute code
                exec(extracted_code)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            #st.write(traceback.print_exc())
            col2.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
            
            
    st.header("Conversational AI")


    #Creating the chatbot interface


    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        df_str = df.to_string()
        st.session_state['past'] = [df_str]

    # We will get the user's input by calling the get_text function
    def get_text():
        input_text = st.text_input("Type your message here: ","",key="input")
        return input_text
    
#     # convert the dataframe into a string
#     df_str = df.to_string()
    
#     st.session_state.past.append(df_str)
    user_input = get_text()

    if user_input:
        output = generate_gpt_reponse(user_input, 1024)
        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
#             message(st.session_state['past'])
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
