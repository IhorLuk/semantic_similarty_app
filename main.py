import streamlit as st

# import to get credentials
import os
from dotenv import load_dotenv

# imports from other files
from text_preprocessing import clean_text
from database_connection import DocumentOpenai, DocumentComments, Session

from scipy.spatial.distance import cosine
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CONNSTR = os.getenv('CONNSTR_POSTGRES')

# init database structure and session
Session = Session(CONNSTR=CONNSTR)

# most important - openAI model
openai = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model='text-embedding-ada-002')

# start the page construction
st.title("Semantic similarity - pgvecotor - testing set 97,000 comments in 14 languages!")

st.text_input(label="text", key="text")

# everything about session - create engine, connect new session and find closest neighbors
Session.create_engine()

def perform_search():
    
    # get query from text field
    query = st.session_state.text
    query = clean_text(query)
    embed_query = openai.embed_query(query)

    Session.connect_session()
    Session.calculate_neighbors(DatabaseDocument=DocumentOpenai, embed_query=embed_query)

    print(query)
    print('\n')

    for neighbor in Session.neighbors:
        comment_text = Session.session.query(DocumentComments).filter(DocumentComments.id == neighbor.id).all()[0].comment_text
        similarity = round(1 - cosine(embed_query, neighbor.embedding), 3)
        id = neighbor.id
        
        #st.text(f"Post id {id}")
        st.text(comment_text)
        st.text(f'Similarity score: {similarity}')

trigger = st.button("Find similar")

if trigger:
    perform_search()


