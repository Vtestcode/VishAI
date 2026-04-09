# RAG Chatbot

A recruiter-facing portfolio chatbot built with FastAPI, OpenAI, Amazon S3, Pinecone, and optional remote MCP tools.

## Architecture

- Source documents live in Amazon S3.
- Embeddings are generated with OpenAI.
- Vector search runs on Pinecone.
- Optional MCP tools can be connected through the OpenAI Responses API.
- Recruiters access the deployed web app and ask questions through the site.


## Features

- Ingest `.txt`, `.md`, and `.pdf` files from S3.
- Store and query embeddings in Pinecone.
- Parse S3 documents with file metadata, section/page labels, and stable chunk IDs.
- Use section-aware chunking plus optional RAPTOR-style summary chunks.
- Incrementally index only changed S3 files using an S3 manifest.
- Translate queries, retrieve candidates, rerank context, and validate answers.
- Answer recruiter-style questions.
- Connect to a remote MCP server and let the assistant call those tools.
- Display connected tools in the chat UI and show which tools were used in a reply.
- Redirect unanswered questions to email when configured.
- Serve both a full-page experience at `/` and an embeddable widget at `/widget`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `CONTACT_EMAIL` | No | Email shown when the answer is not covered by the documents |
| `MODEL_NAME` | No | OpenAI chat model, default `gpt-4o-mini` |
| `REASONING_MODEL_NAME` | No | OpenAI Responses API model for MCP-enabled turns, default `gpt-4.1-mini` |
| `ENABLE_MCP` | No | Enable remote MCP integration, default `false` |
| `MCP_SERVER_URL` | No | Remote MCP server URL, typically the SSE endpoint such as `https://your-server/sse` |
| `MCP_SERVER_LABEL` | No | Short name shown in the UI for the connected MCP server |
| `MCP_SERVER_DESCRIPTION` | No | Optional MCP server description passed to OpenAI |
| `MCP_REQUIRE_APPROVAL` | No | MCP approval policy passed to OpenAI, default `never` |
| `MCP_ALLOWED_TOOLS` | No | Optional comma-separated allowlist of MCP tool names |
| `REDIS_URL` | No | Redis connection URL, used when available for caching |
| `REDISCLOUD_URL` | No | Fallback Redis connection URL for Redis Cloud / older add-ons |
| `REDIS_CACHE_TTL_SECONDS` | No | Cache TTL for MCP tool discovery, default `300` |
| `TOOL_ANSWER_CACHE_TTL_SECONDS` | No | Cache TTL for direct tool-routed answers, default `180` |
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

## Connect A Remote MCP Server

1. Set these environment variables:

```env
ENABLE_MCP=true
MCP_SERVER_URL=https://mcp-server-rag-219ad446fdd2.herokuapp.com/sse
MCP_SERVER_LABEL=rag_tools
MCP_REQUIRE_APPROVAL=never
```

2. Redeploy the app.
3. Open the chatbot UI. If the MCP server is reachable, the chat panel will show a `Connected tools` section.
4. Call `GET /tools` to verify which tools the chatbot can see.

Example:

```powershell
Invoke-RestMethod -Method Get -Uri "https://your-app.herokuapp.com/tools"
```

Notes:
- Most remote MCP servers expose an SSE endpoint such as `/sse`, not just the site root.
- The chatbot keeps using RAG for document context; MCP tools are added as optional extra capabilities on top.

## Redis Caching

If you have Heroku Redis or Redis Cloud attached, set either `REDIS_URL` or `REDISCLOUD_URL`.

The app currently uses Redis to cache:
- MCP tool discovery results
- direct tool-only answers for routed queries such as web search, time, GitHub code search, and README exploration

## Ingestion

Incrementally index changed files from S3:

```powershell
Invoke-RestMethod -Method Post -Uri "https://your-app.herokuapp.com/ingest"
```

Force a full Pinecone namespace rebuild:

```powershell
Invoke-RestMethod -Method Post -Uri "https://your-app.herokuapp.com/ingest?rebuild=true"
```
