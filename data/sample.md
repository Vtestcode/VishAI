# Sample Knowledge Base Document

## About This Project

This is a **RAG (Retrieval-Augmented Generation) Chatbot** built with FastAPI,
LangChain, ChromaDB, and OpenAI.

RAG works by combining a retrieval step with a generation step:

1. **Ingestion** — Documents are loaded, split into chunks, embedded using
   OpenAI's embedding model, and stored in a Chroma vector database.
2. **Retrieval** — When a user asks a question, the system finds the most
   relevant chunks using similarity search.
3. **Generation** — The retrieved chunks are passed as context to GPT-4o-mini
   (or another OpenAI model), which generates a grounded answer.

## Supported File Types

The ingestion pipeline supports:
- **Plain text** (`.txt`)
- **PDF** (`.pdf`)
- **Markdown** (`.md`)

Place your files in the `data/` directory and call the `/ingest` endpoint
(or run `python -m app.rag.ingest`) to rebuild the vector store.

## Deployment

The application is designed to be deployed on **Heroku**:
- A `Procfile` is included for the Heroku Python buildpack.
- Environment variables (`OPENAI_API_KEY`, etc.) should be set via the Heroku dashboard or CLI.

It can also be embedded in **Google Sites** using the `/widget` route in an iframe.

## FAQ

**Q: How many documents can I ingest?**
A: There is no hard limit. ChromaDB stores vectors on disk, and the main
constraint is embedding cost via the OpenAI API.

**Q: Can I use a different model?**
A: Yes. Set the `MODEL_NAME` environment variable to any OpenAI chat model
(e.g., `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`).

**Q: How do I update the knowledge base?**
A: Add, remove, or modify files in the `data/` directory, then call
`POST /ingest` to rebuild the index.
