# Reference for Learning RAG Pipeline Development

This README is a practical guide for building your intuition and skills in Retrieval-Augmented Generation (RAG). It helps you understand foundational concepts and connect them into a working RAG pipeline.

If you're new, start with the basics:

- **Machine learning fundamentals:** vectors, matrices, embeddings, similarity metrics
- **Python fundamentals:** data structures, functions, basic file I/O
- **Deep learning basics:** neural networks, activation functions, loss functions, optimizers

You can run example code without mastering these, but understanding them helps you debug and improve RAG systems.

---

## 1. Vector Databases, Vector Search, and Embeddings

### Core idea
- A **relational database** stores data in rows and columns.
- A **vector database** stores data as high-dimensional points (vectors).

An **embedding** is a numerical representation of an object (text, image, audio, etc.) that captures meaning. Similar items end up closer together in vector space, which enables similarity search.

### Simple analogy
- Relational DB: a 2D table of values  
- Vector DB: a 3D (or higher-dimensional) space of points

When you store vectors, you’re storing **points**, not rows/columns. For example, an image of a dog could be represented by latent dimensions like “four-legged”, “mammal”, and “domestic”—each dimension contributes to where it lands in vector space.

### Why embeddings matter
- Compare **meaning**, not just exact keywords
- Enable **similarity search** (nearest neighbors)
- Power **retrieval** for RAG

---

## 2. RAG Pipeline Core Concepts

### High-level architecture
A typical RAG system has two phases:

1. **Indexing (offline / batch):** prepare your knowledge base for retrieval  
2. **Retrieval + Generation (online):** retrieve relevant context and generate an answer

### Components (what each piece does)

- **Document Corpus (Knowledge Base)**  
  Your source of truth: web pages, PDFs, internal docs, tickets, wikis, etc.

- **Document Loader (Ingestion)**  
  Converts raw sources into text + metadata (title, URL, page number, date, etc.)

- **Text Splitter (Chunking)**  
  Splits text into smaller units (“chunks”) that are easier to embed and retrieve.

- **Embedding Model**  
  Converts chunks (and queries) into vectors.

- **Vector Store / Vector Database**  
  Stores vectors + metadata and supports nearest-neighbor search.

- **Retriever**  
  Given a query, returns top-k relevant chunks. May also apply filters, reranking, or query expansion.

- **Prompt Template**  
  Formats the question + retrieved context into a structured input for the LLM.

- **LLM (Generator)**  
  Produces the final answer using the retrieved context (ideally with citations).

- **Orchestrator (Pipeline/Chain)**  
  Coordinates steps: load → split → embed → store → retrieve → prompt → generate.  
  This is where **LangChain** (or similar) fits.

> Note: In LangChain, many of these are modular components that you connect together.

---

### Indexing phase (offline)

#### 1) Document loading
**Goal:** Turn messy sources into clean text + metadata.

Common loaders:
- **PDF:** PyMuPDF, pdfplumber, PDFMiner
- **HTML/Web:** BeautifulSoup, lxml, newspaper3k (optional), custom scrapers
- **CSV:** pandas, Python `csv`
- **JSON:** Python `json`, jq-like transforms, custom parsing

**Tip:** Always capture metadata (source URL, doc title, page number, timestamp).  
Metadata becomes crucial for filtering and citations later.

#### 2) Cleaning & normalization
Typical steps:
- Remove headers/footers (especially PDFs)
- Fix weird whitespace / hyphenation / broken lines
- Convert tables carefully (tables can break retrieval if flattened poorly)
- Normalize encoding, remove boilerplate

#### 3) Chunking (text splitting)
**Goal:** Retrieve *just enough* relevant context without losing meaning.

Common splitting strategies:
- **By characters (fixed size):** simplest, but risks breaking sentences
- **By paragraphs/sentences:** better readability and context integrity
- **Recursive splitting:** split by sections → paragraphs → sentences → characters (best general approach)

Important chunking parameters:
- **Chunk size:** too small = lacks context; too large = retrieval becomes noisy  
- **Chunk overlap:** helps preserve continuity between chunks  
- **Structure-aware splitting:** preserve headings and sections when possible

---

### Retrieval + generation phase (online)

#### 1) Query understanding
Before retrieval, many systems do some combination of:
- **Query rewriting:** make the user question clearer / more searchable
- **Decomposition:** split a complex question into sub-questions
- **Intent classification:** decide if retrieval is needed or not

#### 2) Retrieval
**Goal:** Fetch top-k chunks that are most relevant.

Retrieval types:
- **Dense retrieval (vector search):** semantic similarity using embeddings
- **Sparse retrieval (keyword/BM25):** exact term matching
- **Hybrid retrieval:** combine dense + sparse for better coverage

Common improvements:
- **Metadata filters:** (e.g., `source=handbook`, `date>2024`, `product=payments`)
- **Reranking:** a second model re-sorts candidates for relevance
- **MMR (diversity):** reduces redundant chunks and increases coverage

#### 3) Augmentation (building the prompt)
**Goal:** Provide the LLM with reliable context.

Good prompt patterns:
- Provide retrieved chunks under a clear “Context” section
- Instruct the model to answer *only using context*
- Ask it to say “I don’t know” when context is insufficient
- Include citation format like: `[source: doc.pdf p.3]`

#### 4) Generation
The LLM generates the answer using:
- The user question
- The retrieved context
- The instruction/prompt template

Common generation controls:
- Short factual answers vs. long explanations
- Citation requirements
- Structured output (JSON, bullets, steps)

---

### Embeddings & vector storage

#### Embedding model options (high-level)
- **Fast / lightweight:** good for prototypes; may lose nuance
- **Balanced:** good tradeoffs for most apps
- **High-quality / expensive:** best performance, higher cost and latency

What matters most in practice:
- Retrieval quality on your actual data
- Latency constraints
- Cost at scale
- Language/domain coverage

#### Vector index strategies (scaling)
- **Flat (brute force):** simplest, accurate; slow at large scale
- **ANN (Approximate Nearest Neighbor):** fast and scalable  
  Common approaches/tools: FAISS, HNSW, Annoy, ScaNN

---

### Evaluation & debugging (core RAG skill)
RAG is usually improved through iteration. Common failure modes:

- **Bad chunking:** retrieved chunks miss the necessary context
- **Poor retrieval:** embeddings don’t match your domain well
- **Context overload:** too many chunks → model gets confused
- **Hallucination:** model answers beyond provided context
- **Source mismatch:** wrong documents retrieved because metadata wasn’t used

Basic evaluation checklist:
- Are retrieved chunks actually relevant to the question?
- Are the chunks complete enough to answer correctly?
- Does the prompt force the model to use context?
- Are citations pointing to the correct sources?

---

## 3. LangChain Notes

LangChain helps implement RAG by providing modular building blocks.

### LangChain components
- **Document Loaders:** ingest PDFs/HTML/CSV/JSON into `Document` objects
- **Text Splitters:** create chunks while maintaining context
- **Embeddings:** plug in an embedding model
- **Vector Stores:** store/query vectors (FAISS, Chroma, etc.)
- **Retrievers:** unify retrieval logic (top-k, filters, hybrid patterns)
- **Chains:** define a sequence of steps for your pipeline
- **Agents:** decide actions/tools dynamically (useful when retrieval is conditional)
- **Memory:** store conversation context (helpful for chat-style RAG)
- **Prompts:** templates that guide consistent and safe LLM output

> Reminder: LangChain simplifies wiring—but you still need to understand chunking, retrieval quality, and evaluation.

---

## Suggested next additions
- A “Mini Projects” section (e.g., PDF Q&A, website FAQ bot, internal docs assistant)
- A “Metrics” section (precision@k, recall@k, answer faithfulness, citation accuracy)
- A “RAG patterns” section (multi-query, reranking, hybrid search, graph RAG)