from langchain.vectorstoes import FAISS 
from langchain.embedding import HugginFaceEmbeddings
from transformers import pipeline 

embeddings  = HuggingFaceEmbeddings(model_name ='senetence-transformers/all-MiniLM-L6-v2')
texts = ["Phising signs: Suspicious URLs, Urgent Language."]
vectorstore = FIASS.from_texts(texts, embeddings)
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_query(query):
    context = vector_store.similarity_search(query)[0].page_content
    return qa(question=query, context=context)['answer']

if __name__ == "__main__":
    query=input("Enter your query: ")
    answer=(answer_query(query))
    