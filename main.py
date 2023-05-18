import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2


#This function will go through pdf and extract and return list of page texts.
def read_and_textify(file):
    pdfReader = PyPDF2.PdfReader (file)
    text_list = []
    #print("Page Number:", len(pdfReader.pages))
    for i in range(len(pdfReader.pages)):
      pageObj = pdfReader.pages[i]
      text = pageObj.extract_text()
      pageObj.clear()
      text_list.append(text)
    file.close()
    return text_list

#LangChain document splitter
# def split_docs(documents,chunk_size=3000,chunk_overlap=100):
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#   docs = text_splitter.create_documents(documents)
#   return docs

# centered page layout
st.set_page_config(layout="centered", page_title="Cooee - Document QA")
st.header("Cooee - Document Question Answering")
st.write("---")

#file uploader
uploaded_file = st.file_uploader("Upload a documents", type=["txt","pdf"])
st.write("---")
if uploaded_file is None:
    st.info(f"""Upload a .pdf file to analyse""")

#Use vectorDB to QnA
elif uploaded_file:
    #get text from documents
    documents = read_and_textify(uploaded_file)
    #text chunking
#     docs = split_docs(documents)
    
    st.write(str(len(documents)) + " document(s) loaded..")
    
    #extract embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
    
    vStore = Chroma.from_texts(docs, embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(docs))])
    st.write(vStore)
    #deciding model
    model_name = "gpt-3.5-turbo"
    # model_name = "gpt-4"
    
    #initiate model
    llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"])
#     model = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=vStore)
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vStore.as_retriever())
    


    st.header("Ask your data")
    user_q = st.text_area("Enter your question here")
    if st.button("Get Response"):
            try:
                # create gpt prompt
#                 result = model.run(user_q)
                result = model({"question": user_q}, return_only_outputs=True)
                st.write(result)
                st.subheader('Your response: {}'.format(' '))
                st.write(result['answer'])
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
