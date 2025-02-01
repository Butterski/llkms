import os
import pickle
from pathlib import Path
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class VectorStoreManager:
    def __init__(self, cache_dir: str = "vector_store_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_path = self.cache_dir / "faiss_index.index"
        self.docs_path = self.cache_dir / "docs.pkl"

    def exists(self) -> bool:
        return self.index_path.exists() and self.docs_path.exists()

    def save(self, vector_store: FAISS):
        faiss.write_index(vector_store.index, str(self.index_path))
        with open(self.docs_path, "wb") as f:
            pickle.dump(vector_store.docstore._dict, f)

    def load(self, embeddings) -> FAISS:
        if not self.exists():
            return None
        index = faiss.read_index(str(self.index_path))
        with open(self.docs_path, "rb") as f:
            docs_dict = pickle.load(f)
        docstore = InMemoryDocstore()
        docstore._dict = docs_dict
        index_to_docstore_id = {i: key for i, key in enumerate(docs_dict.keys())}
        vector_store = FAISS(
            embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        return vector_store
