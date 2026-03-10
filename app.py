import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.title("Multi Document RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload multiple PDFs",
    type="pdf",
    accept_multiple_files=True
)

documents = []

if uploaded_files:

    for file in uploaded_files:

        with open(file.name, "wb") as f:
            f.write(file.read())

        loader = PyPDFLoader(file.name)
        docs = loader.load()

        documents.extend(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        retriever=vectorstore.as_retriever()
    )

    question = st.text_input("Ask question from documents")

    if question:
        answer = qa_chain.run(question)
        st.write(answer)
