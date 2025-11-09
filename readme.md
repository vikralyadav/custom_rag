# 🧠 Custom RAG Agent using LangGraph + Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline powered by **LangGraph**, **LangChain**, and **Ollama** (running the Mistral model locally).  
It demonstrates how to build a fully functional **multi-node AI reasoning graph** that can retrieve, grade, rewrite, and generate answers based on Lilian Weng’s blog content.

---

## 🚀 Overview

The agent performs the following steps:

1. **Web Scraping** – Fetches blog articles using `cheerio`.
2. **Document Splitting** – Chunks large documents into smaller segments using `RecursiveCharacterTextSplitter`.
3. **Vectorization** – Embeds text chunks using `OllamaEmbeddings` and stores them in an in-memory vector store.
4. **Retriever Tool** – Creates a retriever tool using `createRetrieverTool`.
5. **LangGraph Workflow** – Defines a reasoning graph consisting of:
   - `generateQueryOrRespond` → generates queries or responses  
   - `retrieve` → fetches relevant docs from the retriever  
   - `gradeDocuments` → grades retrieved docs for relevance  
   - `rewrite` → rewrites unclear questions  
   - `generate` → generates the final answer  
6. **Ollama Model Integration** – Uses the `mistral` model locally for all text generation steps.

---

##  Architecture



│ LangGraph Workflow │
│ │
│ HumanMessage → generateQueryOrRespond → shouldRetrieve? │
│ ↘ retrieve → gradeDocuments → rewrite/generate │
│ ↓ │
│ END │





---

## 🛠️ Tech Stack

| Component | Library / Tool |
|------------|----------------|
| **LLM** | [Ollama](https://ollama.ai) (`mistral` model) |
| **Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **LangChain Modules** | `@langchain/core`, `@langchain/community`, `@langchain/classic` |
| **Embeddings** | `OllamaEmbeddings` |
| **Vector Store** | `MemoryVectorStore` |
| **Web Loader** | `cheerio` + `node-fetch` |
| **Schema Validation** | `zod` |
| **Runtime** | Node.js v18+ (tested on v22.14.0) |

---

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/vikralyadav/custom_rag
cd custom_rag




---

## 🛠️ Tech Stack

| Component | Library / Tool |
|------------|----------------|
| **LLM** | [Ollama](https://ollama.ai) (`mistral` model) |
| **Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **LangChain Modules** | `@langchain/core`, `@langchain/community`, `@langchain/classic` |
| **Embeddings** | `OllamaEmbeddings` |
| **Vector Store** | `MemoryVectorStore` |
| **Web Loader** | `cheerio` + `node-fetch` |
| **Schema Validation** | `zod` |
| **Runtime** | Node.js v18+ (tested on v22.14.0) |

---
npm install
ollama pull mistral
ollama pull all-minilm
touch .env



#Run the agent 
node agent.js




If everything is set up correctly, you’ll see logs like:
Loaded 3 documents
Split into 120 documents
Retriever tool created: retrieve_blog_posts
Output from node: 'generateQueryOrRespond'
Output from node: 'retrieve'
Output from node: 'gradeDocuments'
Output from node: 'generate'




Example Query

Example input handled by the pipeline:

"What does Lilian Weng say about types of reward hacking?"


Example (expected) output:

Reward hacking can be categorized into two types:
environment or goal misspecification, and reward tampering.



License

This project is open source under the MIT License.


