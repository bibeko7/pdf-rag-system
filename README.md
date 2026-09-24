# 📄 PDF RAG System

An end-to-end **Retrieval-Augmented Generation (RAG)** application for intelligent question answering over PDF documents.

The system processes PDF documents, splits their content into meaningful chunks, generates embeddings, stores and retrieves relevant information, and uses a Large Language Model (LLM) to generate context-aware answers.

The project also includes **evaluation pipelines and automated tests** to assess different components of the RAG system.

---

## 🚀 Features

- 📄 PDF document ingestion
- ✂️ Intelligent document chunking
- 🧠 Text embeddings
- 🔎 Semantic document retrieval
- 🤖 LLM-powered question answering
- 🔗 End-to-end RAG pipeline
- 💾 Vector store integration
- 📊 Retriever evaluation
- 📈 Pipeline evaluation
- 🧪 LLM evaluation
- 🧪 Automated unit tests
- 🖥️ Interactive Streamlit interface
- ⚙️ Configurable application settings

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with Large Language Models.

Instead of asking an LLM to answer a question only from its pretrained knowledge, the system first retrieves relevant information from the provided documents.

### RAG Workflow

```text
              PDF Document
                   │
                   ▼
            Document Ingestion
                   │
                   ▼
              Text Extraction
                   │
                   ▼
             Text Chunking
                   │
                   ▼
              Embeddings
                   │
                   ▼
              Vector Store
                   │
                   ▼
          Relevant Documents
              Retrieved
                   │
                   ▼
             User Question
                   │
                   ▼
          RAG / LLM Pipeline
                   │
                   ▼
            Generated Answer
