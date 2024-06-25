<<<<<<< HEAD
import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# Set the environment variable for the API key
os.environ['GROQ_API_KEY'] = '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'

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

file_path = ("H:\Chatbot\data\Booklet- Laws relating to Women_0.pdf")
loader = PyPDFLoader(file_path)
document=loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size =500, chunk_overlap= 50)
texts = text_splitter.split_documents(document)
embedding_function =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs ={'device':'cpu'})
db=FAISS.from_documents(texts,embedding_function)
DB_FAISS_Path="vectorstores/db_faiss"


def query_vector_db(query):
    # Embed the query
    # Perform the search
    results = db.similarity_search(query, k=5)  #Change k to the number of results you want
    return results

# Function to get response from Groq API

def get_bot_response(user_input):
    relevant_snippet = query_vector_db(user_input)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Based on the following document snippet, answer the question and also mention the meta data associated with the content: '{relevant_snippet}'\nQuestion: {user_input}",
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content


# Streamlit app layout
st.title("Chat with Bot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Display chat messages
for chat in st.session_state.history:
    if chat['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(chat['content'])
    else:
        with st.chat_message("bot"):
            st.markdown(chat['content'])

# User input
user_input = st.chat_input("You:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    bot_response = get_bot_response(user_input)
    st.session_state.history.append({"role": "bot", "content": bot_response})
    
    # Display new messages
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("bot",avatar="🤖"):
        st.markdown(bot_response)
=======
import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Set the environment variable for the API key
os.environ['GROQ_API_KEY'] = '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'

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

file_path = ("H:\Chatbot\data\Booklet- Laws relating to Women_0.pdf")
loader = PyPDFLoader(file_path)
document=loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size =500, chunk_overlap= 50)
texts = text_splitter.split_documents(document)
embedding_function =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs ={'device':'cpu'})
db=FAISS.from_documents(texts,embedding_function)

def query_vector_db(query):
    # Embed the query


    # Perform the search
    results = db.similarity_search(query, k=5)  #Change k to the number of results you want

    return results


# Function to get response from Groq API
def get_bot_response(user_input):
    relevant_snippet = query_vector_db(user_input)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Based on the following document snippet, answer the question: '{relevant_snippet}'\nQuestion: {user_input}",
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content


# Streamlit app layout
st.title("Chat with Bot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Display chat messages
for chat in st.session_state.history:
    if chat['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(chat['content'])
    else:
        with st.chat_message("bot"):
            st.markdown(chat['content'])

# User input
user_input = st.chat_input("You:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    bot_response = get_bot_response(user_input)
    st.session_state.history.append({"role": "bot", "content": bot_response})
    
    # Display new messages
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("bot",avatar="🤖"):
        st.markdown(bot_response)
>>>>>>> 11bb6b364243b5b0398b74a92c2da540688b395d
