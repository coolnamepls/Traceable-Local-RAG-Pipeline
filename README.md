# Traceable Local RAG Pipeline

A local retrieval-augmented generation (RAG) pipeline for structured documents, designed around **traceability, inspectability, and modularity**.

The project combines document parsing, structure-aware chunking, local vector search, reranking, and a small/local language model (SLM). Its main focus is not only generating an answer, but preserving the intermediate artefacts needed to investigate **why** that answer was produced.

> **Project status:** experimental / portfolio project. The current implementation is designed for local execution and iterative development rather than production deployment.

## Why this project?

A typical RAG demo shows the final answer but makes it difficult to inspect failures. If the system retrieves the wrong passage, misses a table, misunderstands a figure, or generates an unsupported answer, it can be difficult to identify where the error entered the pipeline.

This project therefore treats traceability as part of the RAG workflow.

For each processed document, the pipeline preserves source elements, chunk text, chunk metadata, page references, the vector-store representation, reranked retrieval outputs, relevance scores, and the final generated response. This creates an inspection path from:

```text
Generated response
        ↓
Retrieved / reranked chunks
        ↓
Chunk ID + page reference + metadata
        ↓
Parsed document elements
        ↓
Original document
```

The aim is not to claim a formal AI audit trail, but to make model behaviour much easier to inspect, debug, compare, and adapt.

---

## Pipeline Overview

At a high level, the pipeline is:

```text
PDF → Docling parsing → structure-aware chunking + VLM figure descriptions
→ embeddings → local Qdrant vector database
→ query retrieval → reranking → local LLM
→ saved response + ranked evidence
```

A separate flowchart PDF in this repository provides the detailed visual architecture of the full pipeline.


---

## 1. Document Parsing and Preservation

`docling_parser.py` handles document conversion, source-element preservation, structure-aware chunking, and figure enrichment.

Docling is used to parse PDFs while retaining document structure. The parser separately preserves:

- extracted text and metadata;
- tables as Markdown plus metadata;
- figures as image files plus metadata;
- page provenance for source elements.

The purpose of saving these intermediate artefacts is to avoid treating document parsing as a hidden preprocessing step. If retrieval later fails, the parsed source can be inspected independently from the embedding and generation stages.

### Figure handling

Figures are handled separately because visual information may not be represented adequately by normal text extraction.

Each detected figure is passed to `Qwen/Qwen2.5-VL-3B-Instruct`, which generates a retrieval-oriented description covering important elements, trends, relationships, axes, and conclusions where enough context is available.

The generated description is combined with the figure caption and stored as a `DocChunk` with Docling metadata. This allows figure-derived information to participate in retrieval alongside normal text and table chunks.

### Structure-aware chunking

Normal document content is chunked using Docling's `HybridChunker`.

The current implementation:

- uses the embedding model tokenizer when calculating chunk size;
- uses a maximum chunk size of 720 tokens;
- repeats table headers where appropriate;
- preserves headings and document metadata;
- attaches page references to saved chunks.

Each chunk is saved independently with both its text representation and metadata, while a combined `total_chunks.json` file is used for downstream vector-database construction.

---

## 2. Embedding and Vector Database

`vec_database_setup.py` converts the processed chunks into embeddings and stores them in a local Qdrant collection.

The current embedding model is:

```text
Qwen/Qwen3-Embedding-0.6B
```

Embeddings are normalised before storage.

Qdrant is configured with cosine distance and stores each chunk as a point containing:

```text
ID
Embedding vector
Chunk text
Chunk metadata
```

The database currently runs locally using `QdrantClient(path=...)`.

Local Qdrant was chosen because it keeps development lightweight while preserving the same core collection, point, payload, and query abstractions that can later be moved towards a server-based Qdrant deployment.

---

## 3. Retrieval, Reranking and Generation

`RAG_main.py` coordinates query-time retrieval and response generation.

### Retrieval

The user query is embedded using the same embedding model used for the document chunks and is searched against the Qdrant collection.

A configurable retrieval limit controls how many candidate chunks are returned.

### Reranking

Retrieved candidates are then evaluated using:

```text
Qwen/Qwen3-Reranker-0.6B
```

The reranker scores each `(query, chunk)` pair and reorders the retrieved candidates by relevance.

The reranker score is treated as a **relevance signal**, not as confidence in the final generated answer.

### Generation

The ranked chunk text is passed to a local language model:

```text
Qwen/Qwen3-4B
```

The current implementation loads the model in 4-bit quantised form for local inference.

An optional `additional` instruction can be prepended to the RAG prompt. This allows the same retrieval pipeline to be reused for downstream tasks such as report generation, structured JSON output, document comparison, or filling fields in another workflow without changing the retrieval code itself.

---

## Traceability and Error Investigation

The main design goal of the project is to preserve enough intermediate information to investigate failures.

For each query, the pipeline creates a result directory containing the reranked chunks used by the language model.

Each saved retrieval result includes:

```text
Chunk ID
Reranker relevance score
Page reference
Chunk content
```

The final generated response is saved in the same result directory.

This makes it possible to investigate questions such as:

- Did the correct source information exist in the parsed document?
- Was the information placed into an appropriate chunk?
- Did the embedding model retrieve that chunk?
- How highly did the reranker rank it?
- Which evidence was passed to the language model?
- Was the failure caused by retrieval or by generation?

Because parsing, retrieval and generation artefacts are separated, changes to one stage can also be evaluated without losing visibility into the others.

---

## Project Structure

```text
project/
├── RAG_main.py               # Retrieval, reranking and generation
├── docling_parser.py         # Parsing, chunking and figure processing
├── vec_database_setup.py     # Embedding and local Qdrant setup
├── README.md
│
└── <document>_folder/
    ├── text/
    ├── tables/
    ├── figures/
    ├── chunks/
    ├── qdrant_db/
    └── results/
```

The generated document folder keeps parsing artefacts, chunks, the local vector database, and query outputs separate so each stage can be inspected independently.


---

## Current Models

| Component | Model |
|---|---|
| Embedding | `Qwen/Qwen3-Embedding-0.6B` |
| Figure understanding | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Reranking | `Qwen/Qwen3-Reranker-0.6B` |
| Response generation | `Qwen/Qwen3-4B` |
| Vector database | Qdrant |
| Document parsing | Docling |

The VLM and response-generation model are currently configured for local GPU inference, with 4-bit quantisation used to reduce memory requirements.

---

## Running the Pipeline

The current project is configured directly inside the Python scripts rather than through a CLI or configuration file.

Set the document and model IDs in `RAG_main.py`:

```python
embed_model_id = "Qwen/Qwen3-Embedding-0.6B"
vlm_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model_id = "Qwen/Qwen3-4B"

file_scanned = "your_document.pdf"
```

For a new document, run the parsing and indexing stages before querying.

For an already processed document, the existing parsed artefacts and Qdrant database can be reused so that parsing and embedding do not need to be repeated.

A query can then be supplied together with an optional task instruction:

```python
query = "What are the main audit risks discussed in the document?"

prompt = (
    "Return a concise report summarising each risk and the "
    "corresponding actions recommended in the source."
)
```

The response and retrieved evidence will be written to the document's `results/` directory.

---

## Why Local Models?

The project deliberately explores whether a useful RAG workflow can be built around relatively small, locally executed models.

This provides several advantages for experimentation:

- document contents and inference can remain local;
- model behaviour can be tested without depending on an external generation API;
- individual components can be replaced independently;
- the effect of embedding, reranking, chunking and prompting choices can be investigated directly;
- hardware and model-size trade-offs become explicit parts of the system design.

The project is therefore as much an experiment in **RAG system design and evaluation** as it is a question-answering application.

---

## Current Limitations and Planned Improvements

The current implementation is intentionally modular but still experimental.

Planned improvements include:

- configuration files or CLI arguments instead of hard-coded script values;
- automated retrieval evaluation using metrics such as Recall@k and MRR;
- hybrid lexical + dense retrieval for exact control references and domain terminology;
- configurable reranker instructions for domain-specific retrieval;
- more explicit source citations in generated answers;
- support for processing and querying multiple documents in one collection;
- automated tests for parsing, chunk creation and retrieval;
- persistent model loading for repeated queries;
- improved exception handling and logging;
- a reproducible `requirements.txt` or environment file;
- clearer separation between raw chunk text and the contextualised representation used for embedding.

---

## Design Principle

The central idea behind this project is simple:

> **A useful RAG system should make it possible to inspect not only its answer, but the path that produced that answer.**

Rather than treating parsing, chunking and retrieval as invisible intermediate steps, the pipeline preserves them as inspectable artefacts. This makes the system easier to debug, evaluate, adapt to new document workflows, and reason about when the generated result is wrong.

---

## Tech Stack

Python · PyTorch · Hugging Face Transformers · Sentence Transformers · Docling · Qdrant · Qwen3 · Qwen2.5-VL
