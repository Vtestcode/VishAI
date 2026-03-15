# RAG Chatbot

A recruiter-facing portfolio chatbot built with FastAPI, OpenAI, Amazon S3, and Pinecone.

## Architecture

- Source documents live in Amazon S3.
- Embeddings are generated with OpenAI.
- Vector search runs on Pinecone.
- Recruiters access the deployed web app and ask questions through the site.

This project is intentionally configured for hosted use. It does not rely on local document storage or a local vector database.

## Features

- Ingest `.txt`, `.md`, and `.pdf` files from S3.
- Store and query embeddings in Pinecone.
- Answer recruiter-style questions about Vishal's portfolio.
- Redirect unanswered questions to email when configured.
- Serve both a full-page experience at `/` and an embeddable widget at `/widget`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `CONTACT_EMAIL` | No | Email shown when the answer is not covered by the documents |
| `MODEL_NAME` | No | OpenAI chat model, default `gpt-4o-mini` |
| `TOP_K` | No | Number of retrieved chunks |
| `CHUNK_SIZE` | No | Chunk size for ingestion |
| `CHUNK_OVERLAP` | No | Chunk overlap for ingestion |
| `AWS_REGION` | Yes | AWS region for S3 |
| `S3_BUCKET` | Yes | S3 bucket containing portfolio documents |
| `S3_PREFIX` | No | Optional S3 prefix/folder |
| `CHAT_LOG_BUCKET` | No | S3 bucket for persisted chat logs, defaults to `S3_BUCKET` |
| `CHAT_LOG_PREFIX` | No | S3 prefix for chat logs, default `chat-logs/` |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Pinecone index name |
| `PINECONE_NAMESPACE` | No | Pinecone namespace, default `rag-docs` |
| `PINECONE_CLOUD` | No | Pinecone cloud, default `aws` |
| `PINECONE_REGION` | No | Pinecone region, default `us-east-1` |

## Deployment Flow

1. Upload portfolio documents to S3.
2. Set the environment variables in Heroku.
3. Deploy the app.
4. Call `POST /ingest` once to rebuild Pinecone from S3.
5. Share the deployed URL with recruiters.

