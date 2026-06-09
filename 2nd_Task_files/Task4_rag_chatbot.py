import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Sample knowledge base
# ----------------------------
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "RAG stands for Retrieval-Augmented Generation.",
    "LangChain helps build context-aware conversational applications.",
    "Vector databases are used for semantic search."
]

# ----------------------------
# Create vector store
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.create_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(docs, embeddings)

# ----------------------------
# Memory
# ----------------------------
memory = ConversationBufferMemory(return_messages=True)

st.title("Context-Aware RAG Chatbot")

query = st.text_input("Ask a question")

if query:
    memory.chat_memory.add_user_message(query)

    retrieved_docs = vector_db.similarity_search(query, k=2)

    answer = "\n".join([doc.page_content for doc in retrieved_docs])

    memory.chat_memory.add_ai_message(answer)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Conversation History")
    for msg in memory.chat_memory.messages:
        st.write(msg)
