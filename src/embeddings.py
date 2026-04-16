import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion import load_documents

os.environ["TRANSFORMERS_OFFLINE"] = "1"

EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss.index")
DOCS_PATH = os.path.join(EMBEDDINGS_DIR, "documents.pkl")
META_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")
MODEL_META_PATH = os.path.join(EMBEDDINGS_DIR, "model_meta.pkl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "model", "all-MiniLM-L6-v2")
)
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
# MODEL_PATH = os.path.join(BASE_DIR, "model", "all-MiniLM-L6-v2")

print("DEBUG MODEL PATH:", MODEL_PATH)  # 👈 VERY IMPORTANT

model = SentenceTransformer(MODEL_PATH)

def is_data_changed(current_map):
    if not os.path.exists(META_PATH):
        return True

    with open(META_PATH, "rb") as f:
        old_map = pickle.load(f)

    return old_map != current_map


def is_model_changed():
    if not os.path.exists(MODEL_META_PATH):
        return True

    with open(MODEL_META_PATH, "rb") as f:
        meta = pickle.load(f)

    return meta.get("model") != MODEL_NAME


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index

def load_or_create_faiss(data_dir):
    documents, current_map = load_documents(data_dir)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # ========================
    # Load old metadata
    # ========================
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            old_map = pickle.load(f)
    else:
        old_map = {}

    # ========================
    # Detect changes
    # ========================
    new_files = []
    updated_files = []

    for file, mtime in current_map.items():
        if file not in old_map:
            new_files.append(file)
        elif old_map[file] != mtime:
            updated_files.append(file)

    # ========================
    # CASE 1: First time build
    # ========================
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("⚡ First-time FAISS index build...")

        filtered_docs = []
        texts = []

        for doc in documents:
            content = str(doc.get("content", "")).strip()
            if content:
                texts.append(content)
                filtered_docs.append(doc)

        if not texts:
            raise ValueError("No valid text found")

        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, INDEX_PATH)

        with open(DOCS_PATH, "wb") as f:
            pickle.dump(filtered_docs, f)

        with open(META_PATH, "wb") as f:
            pickle.dump(current_map, f)

        print("FAISS index created")
        return index, filtered_docs

    # ========================
    # CASE 2: File updated → FULL REBUILD
    # ========================
    if updated_files:
        print("\n Detected modified files. Rebuilding full index...")
        print("Updated files:", updated_files)

        filtered_docs = []
        texts = []

        for doc in documents:
            content = str(doc.get("content", "")).strip()
            if content:
                texts.append(content)
                filtered_docs.append(doc)

        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, INDEX_PATH)

        with open(DOCS_PATH, "wb") as f:
            pickle.dump(filtered_docs, f)

        with open(META_PATH, "wb") as f:
            pickle.dump(current_map, f)

        print("Rebuilt FAISS index after modification")
        return index, filtered_docs

    # ========================
    # CASE 3: Only NEW files → Incremental add
    # ========================
    print("Loading existing FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        existing_docs = pickle.load(f)

    if new_files:
        print(f"\nNew files detected: {new_files}")

        new_docs = []
        new_texts = []

        for doc in documents:
            if doc.get("source") in new_files:
                content = str(doc.get("content", "")).strip()
                if content:
                    new_docs.append(doc)
                    new_texts.append(content)

        if new_texts:
            print(f"Embedding {len(new_texts)} new chunks...")

            new_embeddings = model.encode(new_texts, show_progress_bar=True)
            new_embeddings = np.array(new_embeddings).astype("float32")

            faiss.normalize_L2(new_embeddings)

            index.add(new_embeddings)

            updated_docs = existing_docs + new_docs

            faiss.write_index(index, INDEX_PATH)

            with open(DOCS_PATH, "wb") as f:
                pickle.dump(updated_docs, f)

            with open(META_PATH, "wb") as f:
                pickle.dump(current_map, f)

            print("Incremental update complete")
            return index, updated_docs

    # ========================
    # CASE 4: No changes
    # ========================
    print("No changes detected. Using existing index.")
    return index, existing_docs