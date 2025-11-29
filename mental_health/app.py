import os
import streamlit as st
from rag import (
    load_documents,
    chunk_documents,
    build_and_save_vector_db,
    load_vector_db,
    generate_answer,
    VECTOR_DIR
)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Mental Health RAG Assistant", layout="wide")
st.title("🧠 Mental Health RAG Assistant")

# ============================================================
# LOAD OR BUILD VECTORSTORE
# ============================================================

if os.path.exists(VECTOR_DIR):
    with st.spinner("📦 Loading vectorstore from disk..."):
        vector_db, embeddings = load_vector_db()
else:
    with st.spinner("⚙️ Building vectorstore (first run)..."):
        docs = load_documents()
        chunks = chunk_documents(docs)
        vector_db, embeddings = build_and_save_vector_db(chunks)
    st.success("Vectorstore created successfully!")

# ============================================================
# CHAT MEMORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []



# Past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("How can I support you today?")

if query:
    # save and display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(query, vector_db)
            st.write(answer)

    # save bot message
    st.session_state.messages.append({"role": "assistant", "content": answer})
