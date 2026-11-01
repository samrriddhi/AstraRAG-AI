import marimo

__generated_with = "0.23.16"

app = marimo.App(width="medium")


with app.setup:
    import marimo as mo

    from pathlib import Path

    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling.document_converter import DocumentConverter
    from fastembed import SparseTextEmbedding, TextEmbedding
    from qdrant_client import QdrantClient, models

    # --------------------------------------------------
    # EMBEDDING MODELS
    # --------------------------------------------------

    embedding_model_name = "BAAI/bge-small-en-v1.5"

    dense_embedder = TextEmbedding(
        model_name=embedding_model_name
    )

    bm25 = SparseTextEmbedding(
        model_name="Qdrant/bm25"
    )

    embedding_dimension = 384

    # --------------------------------------------------
    # LOCAL QDRANT
    # --------------------------------------------------

    qdrant = QdrantClient(
        path="./qdrant_storage"
    )

    # --------------------------------------------------
    # DOCUMENT PROCESSING
    # --------------------------------------------------

    converter = DocumentConverter()


# ==================================================
# 1. USER INPUT
# ==================================================

@app.cell
def _():
    app_form = mo.md(
        """
        # AstraRAG AI - Document Ingestion

        Upload documents to build your searchable knowledge base.

        **1. Select documents**

        {file_upload}

        **2. Select chunking strategy**

        {chunking_select}

        **3. Collection name**

        {collection_name}

        **4. Submit**
        """
    ).batch(
        file_upload=mo.ui.file_browser(
            multiple=True,
            selection_mode="file",
            limit=100,
        ),
        chunking_select=mo.ui.dropdown(
            options=["Hybrid", "Hierarchical"],
            value="Hybrid",
            full_width=True,
            label="Chunking Strategy",
        ),
        collection_name=mo.ui.text(
            value="astra-rag",
            full_width=True,
            label="Collection Name",
        ),
    ).form(bordered=False)

    app_form

    return (app_form,)


# ==================================================
# 2. DOCUMENT LOADING
# ==================================================

@app.cell
def _(app_form):
    documents = []

    mo.stop(not app_form.value)

    uploaded_files = app_form.value["file_upload"]

    if not uploaded_files:
        raise ValueError("Please select at least one document.")

    for uploaded_file in mo.status.progress_bar(uploaded_files):
        file_path = Path(uploaded_file.path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Uploaded file not found: {file_path}"
            )

        result = converter.convert(str(file_path))

        documents.append(
            {
                "document": result.document,
                "source": file_path.name,
            }
        )

    return (documents,)


# ==================================================
# 3. DOCUMENT CHUNKING
# ==================================================

@app.cell
def _(app_form, documents):
    chunks = []

    mo.stop(not app_form.value)

    if app_form.value["chunking_select"] == "Hybrid":
        chunker = HybridChunker()
    else:
        chunker = HierarchicalChunker()

    for item in documents:
        document = item["document"]
        source = item["source"]

        for chunk in chunker.chunk(dl_doc=document):
            content = chunker.contextualize(chunk=chunk)

            if content.strip():
                chunks.append(
                    {
                        "content": content,
                        "source": source,
                    }
                )

    if not chunks:
        raise ValueError(
            "No text chunks were generated from the uploaded documents."
        )

    return (chunks,)


# ==================================================
# 4. EMBEDDING FUNCTIONS
# ==================================================

@app.function
def dense_embed(_chunk: str) -> list[float]:
    embedding = next(iter(dense_embedder.embed([_chunk])))

    return embedding.tolist()


@app.function
def sparse_embed(_chunk: str) -> models.SparseVector:
    response = next(iter(bm25.query_embed(_chunk)))

    return models.SparseVector(
        indices=response.indices.tolist(),
        values=response.values.tolist(),
    )


# ==================================================
# 5. CREATE COLLECTION
# ==================================================

@app.cell
def _(app_form):
    mo.stop(not app_form.value)

    collection_name = app_form.value["collection_name"].strip()

    if not collection_name:
        raise ValueError("Collection name cannot be empty.")

    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=embedding_dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
    )

    return (collection_name,)


# ==================================================
# 6. GENERATE QDRANT POINTS
# ==================================================


@app.cell
def _(chunks):
    points = []

    for index, _chunk in enumerate(chunks, start=1):
        _dense_vector = dense_embed(_chunk["content"])
        _sparse_vector = sparse_embed(_chunk["content"])

        points.append(
    models.PointStruct(
        id=index,
        vector={
            "dense": _dense_vector,
            "sparse": _sparse_vector,
        },
        payload={
            "content": _chunk["content"],
            "source": _chunk["source"],
        },
    )
)

    return (points,)
# ==================================================
# 7. INSERT POINTS IN BATCH
# ==================================================

@app.cell
def _(collection_name, points):
    qdrant.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

    return


# ==================================================
# 8. VERIFY DOCUMENT COUNT
# ==================================================

@app.cell
def _(collection_name, points):
    count_result = qdrant.count(
        collection_name=collection_name,
        exact=True,
    )

    assert count_result.count == len(points), (
        f"Expected {len(points)} points, "
        f"but found {count_result.count}."
    )

    count_result

    return


# ==================================================
# 9. DENSE RETRIEVAL TEST
# ==================================================

@app.cell
def _(collection_name):
    query = "What is this document about?"

    query_vector = next(
        iter(dense_embedder.embed([query]))
    ).tolist()

    dense_results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="dense",
        limit=3,
        with_payload=True,
    )

    dense_results.points

    return


# ==================================================
# 10. HYBRID RETRIEVAL TEST
# ==================================================

@app.cell
def _(collection_name):
    search_query = "What is this document about?"

    dense_vector = next(
        iter(dense_embedder.embed([search_query]))
    ).tolist()

    sparse_vector = next(
        iter(bm25.query_embed(search_query))
    )

    hybrid_results = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=5,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
                using="sparse",
                limit=5,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        limit=3,
        with_payload=True,
    )

    hybrid_results.points

    return


if __name__ == "__main__":
    app.run()
