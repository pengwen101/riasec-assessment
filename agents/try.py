import os
import re
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import pdfplumber

QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "riasec_index_2"
EMBEDDING_DIM   = 1024


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    General multi-column aware extraction using pdfplumber.
    Detects column layout per page dynamically instead of hardcoding splits.
    """
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            width  = page.width
            height = page.height
            top    = height * 0.08
            bottom = height * 0.92

            # Detect columns by finding the largest vertical gap in word positions
            words = page.crop((0, top, width, bottom)).extract_words()
            if not words:
                continue

            # Get all unique x-midpoints of words
            x_positions = sorted(set(round(w['x0']) for w in words))

            # Find the biggest horizontal gap — that's the column separator
            gaps = [(x_positions[i+1] - x_positions[i], x_positions[i])
                    for i in range(len(x_positions) - 1)]
            max_gap, gap_x = max(gaps, key=lambda g: g[0]) if gaps else (0, width)

            # Only split into columns if the gap is significant (>10% of page width)
            if max_gap > width * 0.10:
                left  = page.crop((0,          top, gap_x + max_gap/2, bottom)) \
                            .extract_text(x_tolerance=2, y_tolerance=3) or ""
                right = page.crop((gap_x + max_gap/2, top, width, bottom)) \
                            .extract_text(x_tolerance=2, y_tolerance=3) or ""
                page_text = "\n\n".join(filter(None, [left, right]))
            else:
                # Single column — extract normally
                page_text = page.crop((0, top, width, bottom)) \
                                .extract_text(x_tolerance=2, y_tolerance=3) or ""

            full_text.append(page_text)

    return "\n\n".join(full_text)


def clean_text(raw: str) -> str:
    """General cleanup — works for any document."""
    # Collapse repeated phrases (multi-column artifact)
    cleaned = re.sub(r'(\b\w[\w\s]{3,40}?:\s*)\1{1,3}', r'\1', raw)
    # Normalize whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    return cleaned.strip()


def build_index(document_path: str, collection_name: str = COLLECTION_NAME) -> VectorStoreIndex:
    embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
    Settings.embed_model = embed_model

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Reload if already exists ──────────────────────────────────────────────
    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        if info.vectors_count > 0:
            print(f"Loading existing index ({info.vectors_count} vectors)...")
            return VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model,
            )

    # ── Extract ───────────────────────────────────────────────────────────────
    print("Extracting text from PDF...")
    raw_text   = extract_text_from_pdf(document_path)
    clean      = clean_text(raw_text)

    # ── Semantic chunking — general, no document-specific rules ───────────────
    print("Semantic chunking...")
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        # How sensitive the splitter is to topic shifts.
        # 95 = only split on large meaning shifts (fewer, bigger chunks)
        # 80 = split more aggressively (more, smaller chunks)
        # 95 is better for structured docs like RIASEC where each section
        # is a coherent self-contained topic
        breakpoint_percentile_threshold=95,
        buffer_size=2,  # smooth out noise by considering 2 sentences at once
    )

    from llama_index.core.schema import Document
    doc = Document(text=clean, metadata={"source": os.path.basename(document_path)})
    nodes = splitter.get_nodes_from_documents([doc])

    print(f"Created {len(nodes)} semantic chunks:")
    for i, node in enumerate(nodes):
        print(f"  Chunk {i+1}: {len(node.text)} chars — {node.text[:60].strip()!r}...")

    # ── Store in Qdrant ───────────────────────────────────────────────────────
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print(f"\nIndex built: {len(nodes)} nodes in Qdrant collection '{collection_name}'.")
    return index


index = build_index("../docs/career-theory-model-holland-20170501.pdf")