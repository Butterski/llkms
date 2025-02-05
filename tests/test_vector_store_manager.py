import pickle

import faiss
import pytest

from llkms.utils.vector_store_manager import VectorStoreManager


class DummyFAISS:
    def __init__(self):
        self.index = faiss.IndexFlatL2(1)

        class DummyDocstore:
            _dict = {"1": "dummy"}

        self.docstore = DummyDocstore()


def dummy_faiss_write_index(index, path):
    with open(path, "w") as f:
        f.write("dummy_index")


def dummy_faiss_read_index(path):
    class DummyIndex:
        pass

    return DummyIndex()


@pytest.fixture(autouse=True)
def patch_faiss(monkeypatch):
    monkeypatch.setattr("faiss.write_index", dummy_faiss_write_index)
    monkeypatch.setattr("faiss.read_index", dummy_faiss_read_index)


def test_vector_store_manager_exists(tmp_path):
    cache_dir = tmp_path / "cache"
    manager = VectorStoreManager(cache_dir=str(cache_dir))
    assert not manager.exists()

    manager.index_path.write_text("dummy_index")
    manager.docs_path.write_bytes(pickle.dumps({"1": "dummy"}))
    assert manager.exists()
