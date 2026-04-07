# MovieMate-Chatbot
NLP movie recommendation chatbot with streamlit ui
#  MovieMate – NLP-Based Movie Recommendation Chatbot (RAG System)

##  Project Overview

MovieMate is an end-to-end Natural Language Processing (NLP) project that implements a **Retrieval-Augmented Generation (RAG)** pipeline to build an intelligent movie recommendation chatbot.

The system allows users to input natural language queries (e.g., *“suggest some romcoms”*, *“action movies after 2010”*) and returns context-aware recommendations by combining:

* Semantic search (FAISS + embeddings)
* Large Language Models (LLMs)
* Structured movie metadata

This project demonstrates how modern AI systems integrate **vector databases + LLMs** to solve real-world recommendation problems.

---

##  Objectives

* Build a conversational movie recommendation system
* Implement semantic search using embeddings
* Integrate a vector database (FAISS)
* Apply RAG (Retrieval-Augmented Generation)
* Deploy an interactive UI using Streamlit

---

##  Core Concepts Used

###  Natural Language Processing (NLP)

Used to interpret user queries and extract intent.

###  Embeddings

Text is converted into dense vector representations using:

 sentence-transformers (all-MiniLM-L6-v2)

These embeddings capture semantic meaning rather than exact words.

---

###  Vector Database (FAISS)

FAISS is used to:

* Store embeddings of movie data
* Perform fast similarity search
* Retrieve the most relevant results

---

###  Retrieval-Augmented Generation (RAG)

Instead of relying only on an LLM, the system:

1. Retrieves relevant movie data
2. Injects it into the prompt
3. Generates grounded responses

---

###  Large Language Model (LLM)

Used for:

* Natural language responses
* Context-aware recommendations

Model used:

* Google Gemini API



##  Implementation Details

### 1️. Data Collection & Preprocessing

* Dataset: TMDB Movie Dataset
* Cleaned missing values
* Extracted relevant features:

  * Title
  * Genre
  * Director
  * Cast
  * Keywords
  * Summary

---

### 2️. Feature Engineering

Each movie is converted into a **rich textual representation**

This improves semantic understanding during embedding.

---

### 3️. Embedding Generation

Using:

python
SentenceTransformer("all-MiniLM-L6-v2")


* Converts text → vectors
* Captures meaning (not just keywords)

---

### 4️. FAISS Index Creation

* Normalized embeddings (cosine similarity)
* Stored using FAISS (IndexFlatIP)
* Enables fast nearest-neighbor search

---

### 5️. Retrieval Mechanism

* User query → embedding
* FAISS returns top-K similar movies
* These results act as context

---

### 6️. Prompt Engineering

The LLM is guided using structured prompts:

* Ensures recommendations are based on retrieved data
* Reduces hallucinations
* Improves relevance

---

### 7️. Chatbot Logic

Implemented in chatbot.py:

* Handles user input
* Calls retrieval function
* Passes context to LLM
* Returns formatted response

---

### 8️. Streamlit Interface

Provides:

* Interactive chat UI
* Real-time responses
* Easy demonstration of system

---

smatch (e.g., “romcom” vs “romantic comedy”)
* Managing large embedding computations efficiently
* Handling API/model compatibility issues
* Ensuring relevant retrieval results





