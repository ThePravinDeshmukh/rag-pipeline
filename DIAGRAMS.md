# RAG Pipeline — Flow & Sequence Diagrams

All diagrams use [Mermaid](https://mermaid.js.org/) and render natively on GitHub.

---

## 1. System Architecture Overview

```mermaid
flowchart TD
    subgraph Client["Client Layer"]
        HTTP["HTTP Client / Browser"]
    end

    subgraph API["FastAPI  (/api/v1)"]
        H[GET /health]
        I[POST /ingest]
        Q[POST /query]
        M[GET /metrics]
        D[DELETE /collection]
    end

    subgraph Orchestration["Orchestration (LangGraph)"]
        PL["RAGPipeline"]
        G["Self-Corrective Graph"]
    end

    subgraph Ingestion
        LD["Loaders\nPDF · TXT · CSV · URL · Dir"]
        CL["Cleaners\nunicode · whitespace"]
        CH["Chunkers\nrecursive split"]
    end

    subgraph Embedding
        EM["Embedder\nOpenAI · HuggingFace"]
        IX["Indexer"]
        VDB[("ChromaDB")]
    end

    subgraph Retrieval
        RT["Retriever\nsimilarity search"]
        PB["Prompt Builder\nRAG template"]
    end

    subgraph LLM["LLM Provider"]
        OAI["OpenAI"]
        ANT["Anthropic"]
    end

    subgraph Evaluation
        MX["Metrics\nrelevance · coverage"]
        MN["Monitor\nSQLite query log"]
    end

    HTTP --> API
    I --> PL
    Q --> PL
    D --> IX
    M --> MN

    PL --> LD --> CL --> CH --> EM --> IX --> VDB
    PL --> G
    G --> RT --> VDB
    RT --> PB
    PB --> OAI
    PB --> ANT
    G --> MX
    G --> MN
```

---

## 2. Document Ingestion Flow

```mermaid
flowchart LR
    subgraph Sources
        F1["📄 PDF"]
        F2["📝 TXT / MD"]
        F3["📊 CSV"]
        F4["🌐 URL"]
        F5["📁 Directory"]
    end

    subgraph "Ingestion Stage"
        direction TB
        A["load_sources()"]
        B["clean_documents()\n• strip null chars\n• collapse whitespace\n• NFKC unicode"]
        C["chunk_documents()\nRecursiveCharacterTextSplitter\nchunk_size · chunk_overlap"]
    end

    subgraph "Embedding Stage"
        direction TB
        D["get_embedder()\nOpenAI text-embedding-3-small\nor HuggingFace"]
        E["add_documents()\n→ ChromaDB"]
    end

    F1 & F2 & F3 & F4 & F5 --> A
    A -->|raw Document list| B
    B -->|filtered + cleaned| C
    C -->|chunks with metadata| D
    D -->|vectors| E
    E -->|"✅ N chunks indexed"| DONE["API Response\n{chunks_indexed: N}"]
```

---

## 3. Self-Corrective RAG Query Flow (LangGraph)

```mermaid
flowchart TD
    START(["▶ START\nquery: str"])

    RETRIEVE["🔍 retrieve\nsimilarity_search(query, top_k)"]

    ROUTE{"docs found?"}

    REWRITE["✏️ rewrite_query\nLLM rewrites the query\nto be more specific"]

    EXCEEDED{"rewrite_count\n≥ MAX_REWRITES?"}

    GENERATE["🤖 generate\nRAG_PROMPT → LLM\n(context + question)"]

    FALLBACK["⚠️ fallback answer\n'No relevant information found'"]

    END_NODE(["⏹ END\nanswer + sources + latency"])

    START --> RETRIEVE
    RETRIEVE --> ROUTE

    ROUTE -->|"docs ≥ 1"| GENERATE
    ROUTE -->|"docs = 0"| EXCEEDED

    EXCEEDED -->|"no → retry"| REWRITE
    EXCEEDED -->|"yes → give up"| GENERATE

    REWRITE --> RETRIEVE

    GENERATE -->|"has context"| END_NODE
    GENERATE -->|"no context"| FALLBACK
    FALLBACK --> END_NODE
```

---

## 4. LangGraph State Machine

```mermaid
stateDiagram-v2
    [*] --> retrieve : initial state\n(query, documents=[], rewrite_count=0)

    retrieve --> generate : documents found
    retrieve --> rewrite  : no documents\n& rewrite_count < MAX

    rewrite --> retrieve  : rewrite_count + 1\nnew query string

    retrieve --> generate : no documents\n& rewrite_count ≥ MAX\n(fallback answer)

    generate --> [*] : final state\n(answer, sources, latency_ms)
```

---

## 5. Sequence Diagram — POST /ingest

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI
    participant PL as RAGPipeline
    participant LD as Loaders
    participant CL as Cleaners
    participant CH as Chunkers
    participant EM as Embedder
    participant IX as Indexer
    participant DB as ChromaDB

    Client->>API: POST /api/v1/ingest\n{sources, chunk_size?, chunk_overlap?}
    API->>PL: ingest(sources, chunk_size, chunk_overlap)

    PL->>LD: load_sources(sources)
    LD-->>PL: List[Document] (raw)

    PL->>CL: clean_documents(raw_docs)
    CL-->>PL: List[Document] (cleaned)

    PL->>CH: chunk_documents(cleaned, chunk_size, chunk_overlap)
    CH-->>PL: List[Document] (chunks)

    PL->>IX: add_documents(chunks)
    IX->>EM: get_embedder()
    EM-->>IX: Embeddings model

    loop for each batch of chunks
        IX->>EM: embed_documents(texts)
        EM->>+EM: OpenAI / HuggingFace API
        EM-->>IX: float vectors
    end

    IX->>DB: collection.add(ids, embeddings, documents, metadatas)
    DB-->>IX: ok
    IX-->>PL: N (chunks stored)

    PL-->>API: N
    API-->>Client: 201 Created\n{chunks_indexed: N, message: "..."}
```

---

## 6. Sequence Diagram — POST /query (happy path)

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI
    participant PL as RAGPipeline
    participant GR as LangGraph
    participant RT as Retriever
    participant DB as ChromaDB
    participant EM as Embedder
    participant PB as PromptBuilder
    participant LM as LLM (OpenAI/Anthropic)
    participant MN as Monitor (SQLite)

    Client->>API: POST /api/v1/query\n{question: "..."}
    API->>PL: query(question)
    PL->>GR: invoke({query, documents=[], answer="", rewrite_count=0})

    Note over GR: Node: retrieve
    GR->>RT: retrieve_documents(query, top_k)
    RT->>EM: embed_query(query)
    EM-->>RT: query vector
    RT->>DB: similarity_search(query_vector, k=top_k)
    DB-->>RT: List[Document, score]
    RT-->>GR: documents (filtered by score_threshold)

    Note over GR: Route: docs found → generate
    GR->>PB: format_context(documents)
    PB-->>GR: context string

    Note over GR: Node: generate
    GR->>LM: RAG_PROMPT | LLM\n{context, question}
    LM-->>GR: AIMessage(content="...")

    GR-->>PL: final_state {answer, documents, query}

    PL->>MN: log_query(query, answer, num_docs, latency_ms)
    MN->>MN: INSERT INTO query_log

    PL-->>API: {answer, sources, latency_ms}
    API-->>Client: 200 OK\n{question, answer, sources, latency_ms}
```

---

## 7. Sequence Diagram — POST /query (self-correction path)

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI
    participant GR as LangGraph
    participant RT as Retriever
    participant DB as ChromaDB
    participant LM as LLM
    participant MN as Monitor

    Client->>API: POST /api/v1/query\n{question: "vague or ambiguous query"}
    API->>GR: invoke(state)

    Note over GR: Node: retrieve (attempt 1)
    GR->>RT: retrieve_documents(original_query)
    RT->>DB: similarity_search(...)
    DB-->>RT: [] (empty — no match)
    RT-->>GR: documents = []

    Note over GR: Route: no docs, rewrite_count=0 < MAX → rewrite
    GR->>LM: rewrite_query(original_query)
    LM-->>GR: rewritten_query (more specific)

    Note over GR: Node: retrieve (attempt 2)
    GR->>RT: retrieve_documents(rewritten_query)
    RT->>DB: similarity_search(...)
    DB-->>RT: [doc1, doc2, ...]
    RT-->>GR: documents (found)

    Note over GR: Route: docs found → generate
    GR->>LM: RAG_PROMPT | LLM\n{context, rewritten_query}
    LM-->>GR: answer

    GR-->>API: final_state
    API->>MN: log_query(...)
    API-->>Client: 200 OK\n{answer, sources, latency_ms}
```

---

## 8. Sequence Diagram — GET /metrics

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI
    participant MN as Monitor (SQLite)

    Client->>API: GET /api/v1/metrics

    API->>MN: get_stats()
    MN->>MN: SELECT COUNT(*), AVG(latency_ms),\nAVG(num_docs), MIN, MAX\nFROM query_log
    MN-->>API: {total_queries, avg_latency_ms, ...}

    API->>MN: get_recent_queries(limit=20)
    MN->>MN: SELECT ... ORDER BY id DESC LIMIT 20
    MN-->>API: List[{id, timestamp, query, answer, ...}]

    API-->>Client: 200 OK\n{stats: {...}, recent_queries: [...]}
```

---

## 9. Component Dependency Graph

```mermaid
flowchart BT
    config["⚙️ config.py\nSettings"]

    loaders["loaders.py"] --> config
    cleaners["cleaners.py"]
    chunkers["chunkers.py"] --> config

    embedder["embedder.py"] --> config
    indexer["indexer.py"] --> config
    indexer --> embedder

    retriever["retriever.py"] --> config
    retriever --> indexer
    prompt_builder["prompt_builder.py"]

    llm["_llm.py"] --> config
    graph["graph.py"] --> retriever
    graph --> prompt_builder
    graph --> llm
    graph --> config

    pipeline["pipeline.py"] --> loaders
    pipeline --> cleaners
    pipeline --> chunkers
    pipeline --> indexer
    pipeline --> graph
    pipeline --> monitor

    metrics["metrics.py"]
    monitor["monitor.py"] --> config

    schemas["schemas.py"]
    routes["routes.py"] --> pipeline
    routes --> monitor
    routes --> indexer
    main["main.py"] --> routes
    main --> config
```
