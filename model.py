from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

DB_FAISS_Path = "vectorstores/db_faiss"

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_Path, embeddings, allow_dangerous_deserialization=True)
    return db

def query_vector_db(query, db):
    # Embed the query
    query_embedding = db.embed_query(query)
    
    # Perform the search
    results = db.similarity_search_by_vector(query_embedding, k=5)  # Change k to the number of results you want
    
    return results

if __name__ == '__main__':
    # Load the FAISS index
    db = load_vector_db()
    
    # Define your query
    query = "What are the various laws for women's safety"
    
    # Perform the query
    results = query_vector_db(query, db)
    
    # Print the results
    for result in results:
        print(result.metadata)  # Adjust based on your needs (e.g., result.page_content, result.metadata)
