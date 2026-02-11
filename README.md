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

## 3. Advanced RAG Techniques

As you move beyond basic RAG, the focus shifts from “making retrieval work” to making it **reliable, precise, and scalable**. Advanced RAG techniques improve retrieval quality, reduce hallucinations, handle complex questions, and support specialized data types (like images, tables, or multiple document granularities).

---

### Key Components

- **Hybrid Retrieval (Dense + Sparse)**  
  Combines vector search (semantic similarity) with keyword search (BM25/lexical).  
  Helpful when queries include exact tokens like IDs, error codes, filenames, or proper nouns.

- **Multi-Step Retrieval (Coarse-to-Fine)**  
  Uses stages to improve scalability and relevance: retrieve broadly first, then refine.  
  Common pattern: **document/section retrieval → chunk retrieval → reranking**.

- **Reranking (Second-Stage Retrieval)**  
  After retrieving candidates (e.g., top-20), rerank them using stronger relevance signals.  
  Improves precision of the final **top-k chunks** used in the prompt.

- **Filtering (Noise Reduction + Control)**  
  Exclude irrelevant or low-quality content before/after retrieval.  
  Often uses content rules (duplicates/boilerplate), metadata constraints (date/source), or user scope.

- **Hierarchical Retrieval**  
  Retrieve at multiple granularities: document → section → chunk.  
  Reduces noise in long documents and improves context targeting.

- **Multi-Vector Retrieval**  
  Store multiple embeddings per unit (chunk/doc) to represent different views: content, title, summary, entities, or Q&A pairs.  
  Improves recall across different query styles.

- **Contextual / Query-Aware Retrieval**  
  Rewrite or expand the query using synonyms, related concepts, or intent-aware rewriting to improve retrieval matches.

- **Knowledge-Enhanced Retrieval**  
  Use structured knowledge like entity linking, taxonomies, or knowledge graphs to connect related concepts and reduce ambiguity.

- **Cross-Modal Retrieval**  
  Retrieve across modalities (text ↔ image ↔ audio).  
  Useful for diagrams, screenshots, tables-as-images, and mixed-media knowledge bases.

- **Active Learning (User Feedback Loop)**  
  Use user feedback (clicks, ratings, corrections) to improve chunking, retrieval, reranking, and prompting over time.
  
---

### Text Splitting and Chunking Strategies

Chunking strongly affects retrieval quality—if the chunk boundaries are wrong, even the best embedding model may retrieve the wrong context.

- **Importance of Text Splitting**  
  Splitting controls what becomes “retrievable.” If key facts are split apart, retrieval may miss them or return incomplete context.

- **Context Preservation**  
  Use overlap, structure-aware splitting (headings/sections), and sentence/paragraph boundaries to keep meaning intact.

- **Impact on Retrieval Accuracy**  
  Good chunking improves both:
  - **Recall:** ability to find relevant information
  - **Precision:** reducing irrelevant results in top-k

---

### Balancing Chunk Size with Semantic Coherence

Chunking is a tradeoff between speed, cost, and answer quality.

- **Chunk Size**
  - **Smaller chunks** → faster retrieval and more precise matches, but may lose necessary context.
  - **Larger chunks** → better context and coherence, but may include irrelevant text and waste tokens.

- **Semantic Coherence**
  The goal is to make each chunk represent a *complete idea*.  
  A good chunk usually contains enough context to answer a question without relying on neighboring chunks.

---

### Techniques for Improving Retrieval Accuracy

- **Synonym Expansion**  
  Expand queries with synonyms or alternate terms.  
  Example: “refund” → “reimbursement”, “chargeback”, “return”.

- **Concept Expansion**  
  Add related concepts that often appear with the same intent.  
  Example: “reset password” → “account recovery”, “2FA”, “email verification”.

- **Query Reformulation (Query Rewriting)**  
  Rewrite the user query to be more searchable and retrieval-friendly.  
  Example: “Why can’t I log in?” → “login error troubleshooting steps and common causes”.

> Tip: Many production systems use a combination: **hybrid retrieval + reranking + good chunking** before trying more complex solutions.

---

### Reranking and Filtering Retrieved Documents

- **Reranking**  
  Reorders retrieved results using additional signals to improve relevance (often applied after retrieving top-k or top-n candidates).

  Common reranking approaches:
  - **Contextual Reranking:** uses the query + candidate chunk together to judge relevance more accurately.
  - **Model-Based Reranking:** uses ML models (e.g., cross-encoders, LLM rerankers) to score and reorder results.
  - **Feedback-Based Ranking:** adapts ranking based on user feedback (clicks, upvotes/downvotes, corrections).

- **Filtering**  
  Excludes irrelevant, low-quality, or out-of-scope data before or after retrieval to reduce noise and improve answer quality.

  Common filtering approaches:
  - **Content-Based Filtering:** remove boilerplate, short/empty chunks, duplicates, or low-information text.
  - **Metadata Filtering:** filter by source, date, department, doc type, language, access level, etc.
  - **User-Preference Filtering:** personalize results based on role, history, or user-selected scope.

---

### Multi-Step Retrieval

- **Coarse-to-Fine Retrieval Strategies**  
  An initial broad search (**coarse**) is followed by refined retrieval (**fine**) to handle large datasets efficiently.

  How it works:
  1. **Coarse stage:** retrieve high-level candidates (documents/sections) quickly.
  2. **Fine stage:** search within those candidates using smaller chunks, better reranking, or stricter filters.

  Advantages:
  - **Scalability:** reduces compute by narrowing the search space early.
  - **Accuracy:** later stages apply deeper analysis (reranking, better chunk matching) to improve relevance.

---

## 4. Caching Responses Module

The `5_caching_responses_rag` folder now follows a step-by-step learning structure:

- `5_caching_responses_rag/a_in_memory/with_openai.py`: in-memory query cache (single-process demo)
- `5_caching_responses_rag/b_postgres/with_openai.py`: persistent query cache in PostgreSQL
- `5_caching_responses_rag/c_redis/with_openai.py`: fast query cache in Redis (optional TTL)

Module guide:

- `5_caching_responses_rag/README.md`

Why this matters:

- Reduces repeated LLM cost for identical query + retrieved context
- Improves response latency for repeated questions
- Shows tradeoffs between local memory, durable SQL cache, and low-latency key-value cache

---

## Suggested next additions
- A “Mini Projects” section (e.g., PDF Q&A, website FAQ bot, internal docs assistant)
- A “Metrics” section (precision@k, recall@k, answer faithfulness, citation accuracy)
- A “RAG patterns” section (multi-query, reranking, hybrid search, graph RAG)

## References
  SQL Agent
    - https://docs.langchain.com/oss/python/langchain/sql-agent
    - https://docs.langchain.com/oss/python/langchain/rag
  Caching Response:
    - https://aws.amazon.com/blogs/database/optimize-llm-response-costs-and-latency-with-effective-caching/
    - https://colab.research.google.com/github/sugarforever/LangChain-Tutorials/blob/main/LangChain_Caching.ipynb
