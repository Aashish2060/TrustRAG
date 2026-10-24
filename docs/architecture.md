# TrustRAG Architecture

```mermaid
flowchart TD
  U[User / Streamlit] --> API[FastAPI or local service]
  API --> ING[PDF/TXT/DOCX extraction]
  ING --> CH[Chunking and metadata]
  CH --> DS[(Local JSON document store)]
  Q[Question] --> RET[TF-IDF retrieval]
  DS --> RET
  RET --> GEN[Extractive or configured API generation]
  GEN --> VER[Lexical claim verifier]
  VER --> REL[Heuristic reliability score]
  REL --> CTRL[Bounded correction controller]
  CTRL --> OUT[Answer / qualified answer]
  OUT --> EVAL[Evaluation exports]
```

## Design notes

- Retrieval is deliberately replaceable; current retrieval is TF-IDF, not dense semantic search.
- Generation is abstracted behind `LLMClient`; local mode extracts relevant sentences and API mode uses an OpenAI-compatible endpoint.
- Verification and reliability are separate modules. The current verifier is lexical and must not be confused with natural-language inference.
- Correction is bounded to at most two attempts and keeps the previous answer if the revision does not meet the current acceptance heuristic.
- Source identifiers point to stored chunks; page numbers are included when extraction supports them.
- The current JSON store is a single-user prototype, not a concurrent production database.
