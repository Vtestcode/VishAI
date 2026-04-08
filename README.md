# RAG Chatbot

A recruiter-facing portfolio chatbot built with FastAPI, OpenAI, Amazon S3, and Pinecone.

## Architecture

- Source documents live in Amazon S3.
- Embeddings are generated with OpenAI.
- Vector search runs on Pinecone.
- Recruiters access the deployed web app and ask questions through the site.


## Features

- Ingest `.txt`, `.md`, and `.pdf` files from S3.
- Store and query embeddings in Pinecone.
- Parse S3 documents with file metadata, section/page labels, and stable chunk IDs.
- Use section-aware chunking plus optional RAPTOR-style summary chunks.
- Incrementally index only changed S3 files using an S3 manifest.
- Translate queries, retrieve candidates, rerank context, and validate answers.
- Answer recruiter-style questions.
- Redirect unanswered questions to email when configured.
- Serve both a full-page experience at `/` and an embeddable widget at `/widget`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `CONTACT_EMAIL` | No | Email shown when the answer is not covered by the documents |
| `MODEL_NAME` | No | OpenAI chat model, default `gpt-4o-mini` |
| `TOP_K` | No | Number of retrieved chunks |
| `RETRIEVAL_CANDIDATE_K` | No | Number of candidates retrieved before reranking, default `15` |
| `QUERY_REWRITE_COUNT` | No | Number of translated retrieval queries, default `3` |
| `CHUNK_SIZE` | No | Chunk size for ingestion |
| `CHUNK_OVERLAP` | No | Chunk overlap for ingestion |
| `ENABLE_RERANKING` | No | Use LLM reranking before answer generation, default `true` |
| `ENABLE_ANSWER_VALIDATION` | No | Validate whether the answer is grounded and responsive, default `true` |
| `ENABLE_RAPTOR` | No | Add RAPTOR-style parent summary chunks during ingest, default `true` |
| `RAPTOR_GROUP_SIZE` | No | Number of leaf chunks per summary chunk, default `6` |
| `AWS_REGION` | Yes | AWS region for S3 |
| `S3_BUCKET` | Yes | S3 bucket containing portfolio documents |
| `S3_PREFIX` | No | Optional S3 prefix/folder |
| `RAG_MANIFEST_KEY` | No | S3 key for incremental indexing manifest, defaults to `.rag-index-manifest.json` under `S3_PREFIX` |
| `CHAT_LOG_BUCKET` | No | S3 bucket for persisted chat logs, defaults to `S3_BUCKET` |
| `CHAT_LOG_PREFIX` | No | S3 prefix for chat logs, default `chat-logs/` |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Pinecone index name |
| `PINECONE_NAMESPACE` | No | Pinecone namespace, default `rag-docs` |
| `PINECONE_CLOUD` | No | Pinecone cloud, default `aws` |
| `PINECONE_REGION` | No | Pinecone region, default `us-east-1` |

## Deployment Flow

1. Upload portfolio documents to S3.
2. Set the environment variables.
3. Deploy the app.
4. Call `POST /ingest` to incrementally add changed S3 files to Pinecone.
5. Share the deployed URL with recruiters.

## Ingestion

Incrementally index changed files from S3:

```powershell
Invoke-RestMethod -Method Post -Uri "https://your-app.herokuapp.com/ingest"
```

Force a full Pinecone namespace rebuild:

```powershell
Invoke-RestMethod -Method Post -Uri "https://your-app.herokuapp.com/ingest?rebuild=true"
```
