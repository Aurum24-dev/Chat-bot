
import os
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


os.environ['GROQ_API_KEY'] = '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'

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

embedding_function =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs ={'device':'cpu'})
DB_FAISS_Path="vectorstores/db_faiss"
new_db = FAISS.load_local( DB_FAISS_Path, embedding_function, allow_dangerous_deserialization=True)

"""query="What are the laws that protect women against domestic violence"
docs = new_db.similarity_search(query, k=1)
print(type(docs))"""


def query_vector_db(query):
    # Embed the query
    # Perform the search
    results = new_db.similarity_search(query, k=5)  #Change k to the number of results you want
    return results

# Function to get response from Groq API

def get_bot_response(user_input):
    relevant_snippet = query_vector_db(user_input)
    relevant_snippets_str = "\n".join([f"Document ID: {item[0]}, Score: {item[1]}" for item in relevant_snippet])
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Based on the following document snippet, answer the question: '{relevant_snippets_str}'\nQuestion: {user_input} also give the meta data if any",
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

if __name__ == '__main__':
  response=get_bot_response("What are the laws that protect women against domestic violence")