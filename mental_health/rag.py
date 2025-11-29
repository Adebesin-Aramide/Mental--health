import os
import glob
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# HuggingFace LLM
from huggingface_hub import InferenceClient

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


DATA_DIR = "data"
VECTOR_DIR = "vectorstore"        
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("❗ Please set HF_TOKEN in your .env file.")

llm_client = InferenceClient(api_key=HF_TOKEN)

def load_documents():
    """Loads .txt documents from data directory."""
    files = glob.glob(f"{DATA_DIR}/*.txt")
    if not files:
        raise ValueError(f"❗ No .txt files found in folder: {DATA_DIR}")

    docs = []
    for f in files:
        loader = TextLoader(f, encoding="utf-8")
        docs.extend(loader.load())

    print(f"📚 Loaded {len(docs)} documents.")
    return docs


def chunk_documents(docs):
    """Splits documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} chunks.")
    return chunks


def build_and_save_vector_db(chunks):
    """Embeds chunks & saves vector DB to disk."""
    print("🔢 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("🗂️ Building FAISS vector store...")
    vector_db = FAISS.from_documents(chunks, embeddings)

    print("💾 Saving vectorstore to disk...")
    vector_db.save_local(VECTOR_DIR)

    print("✅ Vector DB saved and ready.")
    return vector_db, embeddings


def load_vector_db():
    """Loads FAISS vector DB from local storage."""
    print("📦 Loading existing FAISS vector DB from disk...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector DB loaded successfully.")
    return vector_db, embeddings


def retrieve_context(query, db, k=5):
    """Retrieve best matching chunks from FAISS DB."""
    retriever = db.as_retriever(search_type='similarity', search_k=k)
    docs = retriever.get_relevant_documents(query)

    print(f"🔍 Retrieved {len(docs)} relevant chunks.")
    context = "\n\n".join([d.page_content for d in docs])
    return context


def call_llm(context, query):
    """Use HF InferenceClient to generate a response."""

    system_prompt = (
        "You are a helpful, empathetic mental health support assistant.\n"
        "You speak gently, validate feelings, ask open-ended questions, and avoid judgment.\n"
        "If a user shows signs of severe distress, advise them to seek professional help.\n"
        "Use simple, friendly language.\n\n"
        "Only use the information from the retrieved notes.\n"
        "If the answer is not present in the notes, say:\n"
        "'I'm not sure based on my notes, but I'm here to support you.'\n\n"
        f"CONTEXT:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = llm_client.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=350,
        temperature=0.3
    )

    return response.choices[0].message["content"]

def generate_answer(query, db):
    context = retrieve_context(query, db)
    return call_llm(context, query)


def main():
    print("===============================================")
    print(" 🤖 RAG CHATBOT FOR MENTAL HEALTH SUPPORT ")
    print("===============================================\n")

    # Check if vectorstore exists
    if os.path.exists(VECTOR_DIR):
        vector_db, embeddings = load_vector_db()
    else:
        # First-time run → build vector DB
        docs = load_documents()
        chunks = chunk_documents(docs)
        vector_db, embeddings = build_and_save_vector_db(chunks)

    print("\nChatbot is READY! Ask your questions.")
    print("Type 'exit' to quit.\n")

    # Chat loop
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Chatbot: Take care 💛")
            break

        answer = generate_answer(query, vector_db)
        print("\nChatbot:", answer, "\n")


if __name__ == "__main__":

    main()

