import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from langchain_community.agent_toolkits import create_sql_agent

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

# app config
st.set_page_config(page_title="Fundus Bot", page_icon="🤖")
st.title("Your Personal Assistant")

engine = create_engine("sqlite:///patientHealthData.db")
db = SQLDatabase(engine=engine)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

logo_path = 'FundusAI_Logo.png'
st.sidebar.image(logo_path, use_column_width=True)

# def get_response(user_query, chat_history):

    # template = """
    # You are a helpful assistant. Answer the following questions considering the history of the conversation:

    # Chat history: {chat_history}

    # User question: {user_question}
    # """

    # template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    # {schema}

    # Question: {question}
    # SQL Query: {query}
    # SQL Response: {response}"""
    # prompt_response = ChatPromptTemplate.from_template(template)

    # prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatOpenAI()
        
    # chain = prompt | llm | StrOutputParser()
    
    # return chain.stream({
    #     "chat_history": chat_history,
    #     "user_question": user_query,
    # })

def get_response(user_query):

    write_query = create_sql_query_chain(llm, db) # Write SQL query based on user input
    execute_query = QuerySQLDataBaseTool(db=db) # Execute the query to get answer


    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    answer = answer_prompt | llm | StrOutputParser()

    chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
    )

    return chain.stream({
        "question": user_query,
    })


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Elfie. I am here to provide insights on your health data"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query))

    st.session_state.chat_history.append(AIMessage(content=response))