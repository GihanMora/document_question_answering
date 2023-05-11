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
from streamlit_chat import message


# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def create_connection(db_name: str) -> Connection:
    conn = sqlite3.connect(db_name)
    return conn

def run_query(conn: Connection, query: str) -> pd.DataFrame:
    df = pd.read_sql_query(query, conn)
    return df

def create_table(conn: Connection, df: pd.DataFrame, table_name: str):
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def generate_gpt_reponse(gpt_input, max_tokens):

    # load api key from secrets
    openai.api_key = st.secrets["openai_api_key"]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=max_tokens,
        temperature=0,
        messages=[
            {"role": "user", "content": gpt_input},
        ]
    )

    gpt_response = completion.choices[0].message['content'].strip()
    return gpt_response


def extract_code(gpt_response):
    """function to extract code and sql query from gpt response"""

    if "```" in gpt_response:
        # extract text between ``` and ```
        pattern = r'```(.*?)```'
        code = re.search(pattern, gpt_response, re.DOTALL)
        extracted_code = code.group(1)

        # remove python from the code (weird bug)
        extracted_code = extracted_code.replace('python', '')

        return extracted_code
    else:
        return gpt_response

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

def generate_response(prompt):
    completions = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message 



# wide layout
st.set_page_config(layout="wide", page_title="Cooee + ChatGPT")

st.header("Ask Cooee(ChatGPT Powered)")
st.write("---")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
st.write("---")
if uploaded_file is None:
    st.info(f"""Upload a .csv file to analyse""")


elif uploaded_file:
    df = pd.read_csv(uploaded_file)


    # Apply the custom function and convert date columns
    for col in df.columns:
        # check if a column name contains date substring
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col])
            # remove timestamp
            #df[col] = df[col].dt.date

    # reset index
    df = df.reset_index(drop=True)

    # replace space with _ in column names
    df.columns = df.columns.str.replace(' ', '_')

    cols = df.columns
    cols = ", ".join(cols)

    with st.expander("Preview of the uploaded CSV file"):
        st.table(df.head())

    conn = create_connection(":memory:")
    table_name = "my_table"
    create_table(conn, df, table_name)
    st.write("---")


                
    col1, col2 = st.columns(2)
    col1.header("Question Answering")
    user_q = col1.text_area("Enter your question here")
    if col1.button("Get Response"):
            try:
                # create gpt prompt
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
