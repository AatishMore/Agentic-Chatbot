 # 🤖 LangGraph Agentic Chatbot

This project is a dynamic chatbot application built using **LangGraph**, **FastAPI**, and **Streamlit**. It supports:

- **RAG (Retrieval-Augmented Generation)** from PDFs
- **Web search** fallback for external queries
- Chat history and context handling
- Dynamic tool selection using LLM reasoning

---

## **Features**

1. Upload PDFs for document-based Q&A.
2. Intelligent chat powered by LangGraph agent and LLMs.
3. Combined internal and external search for more accurate responses.
4. Clean, interactive frontend built with Streamlit.
5. Object-oriented design for maintainability.

---

## **Project Structure**
chatbot_class_object/
│
├─ fastapi_app.py # FastAPI backend endpoints
├─ streamlit_app.py # Streamlit frontend UI
├─ agent.py # LangGraph agent logic
├─ tools/
│ ├─ rag_tool.py # RAG pipeline & PDF indexing
│
├─ temp_docs/ # Temporary uploaded PDFs
├─ requirements.txt # Python dependencies


1] Create a virtual environment and install dependencies
2] Run FastAPI backend
3] Run Streamlit frontend

Usage:
Open the Streamlit app in your browser.
Upload PDFs for internal document search.
Ask questions via chat input.
The bot will dynamically decide whether to use RAG, web search, or direct LLM generation.
Chat history and tool usage will be displayed in the UI.

Note:
Both FastAPI and Streamlit must run simultaneously for the app to work.
