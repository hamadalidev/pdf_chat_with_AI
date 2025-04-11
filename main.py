# Import necessary libraries
import os
import argparse
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def create_vector_db(text, embeddings_model, pdf_path):
    """Create a vector database with improved chunking strategy."""
    # Determine appropriate chunk size based on text length
    total_length = len(text)

    if total_length < 50000:  # Short document
        chunk_size = 1000
        chunk_overlap = 200
    elif total_length < 200000:  # Medium document
        chunk_size = 1500
        chunk_overlap = 300
    else:  # Long document
        chunk_size = 2000
        chunk_overlap = 400

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    print(f"Document split into {len(chunks)} chunks")

    # Create metadata for each chunk
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": pdf_path,
            "chunk_id": i,
            "chunk_total": len(chunks),
        }
        metadatas.append(metadata)

    # Create vector database with metadata
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings_model,
        metadatas=metadatas
    )

    return vectorstore


def create_qa_chain(vectorstore, llm):
    """Create a conversational retrieval chain."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return qa_chain


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chat with a PDF using a local AI model')
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file')
    args = parser.parse_args()

    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        return

    print("Loading and processing PDF...")
    text = extract_text_from_pdf(args.pdf_path)

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating vector database from PDF content...")
    vectorstore = create_vector_db(text, embeddings, args.pdf_path)

    print("Setting up language model...")
    # Use a model that supports text-generation task
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token="token",# More compatible model
        task="text-generation",
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 512,
            "top_p": 0.95
        }
    )

    print("Creating QA chain...")
    qa_chain = create_qa_chain(vectorstore, llm)

    print("\n--- PDF Chat Ready ---")
    print("Type 'exit' to end the conversation.\n")

    # Simple chat loop
    while True:
        query = input("You: ")

        if query.lower() == 'exit':
            break

        print("AI: ", end="", flush=True)
        try:
            # Fix the deprecation warning by using invoke instead of __call__
            result = qa_chain.invoke({"question": query})
            print(result["answer"])
        except Exception as e:
            print(f"Error: {e}")

    print("Conversation ended.")


if __name__ == "__main__":
    main()
