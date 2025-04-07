# PDF Chat with AI

This program allows you to interact with a PDF document using an AI model. It extracts text from the provided PDF file, splits the content into manageable chunks, and creates a vector database to enable conversational queries. The AI model will respond to your questions based on the content in the PDF.

## Features

- Extract text from PDF using PyMuPDF.
- Split text into chunks with an intelligent chunking strategy.
- Store the chunks in a vector database for fast retrieval.
- Use a conversational AI model to answer questions based on the PDF content.
- Memory management to hold context of the conversation.

## Requirements

- Python 3.7 or higher
- The following Python libraries:
  - `fitz` (PyMuPDF) - for PDF text extraction
  - `langchain` - for managing LLMs, vector stores, and conversational chains
  - `faiss-cpu` or `faiss-gpu` - for vector database management
  - `sentence-transformers` - for embeddings

You can install the necessary libraries with the following command:

```bash
pip install -r requirements.txt


## How you run this.
```bash
python main.py pdf_file_path
