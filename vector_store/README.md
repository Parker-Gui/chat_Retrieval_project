# Vector storage

Step 5 uses Neo4j's native vector index through the official `neo4j-graphrag`
package. Embeddings are stored on `KnowledgeChunk.embedding`; no separate FAISS
or Chroma files are required in this directory.

The embedding model cache is downloaded to the ignored `tools/model_cache/`
directory. The reproducible experiment configuration is stored in
`experiments/midterm/configs/vector_rag.json`.
