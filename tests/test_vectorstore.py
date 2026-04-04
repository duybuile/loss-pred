import pytest


def test_init_uses_langchain_chroma_backend(monkeypatch):
    from app import vectorstore as vectorstore_module

    calls = {"count": 0}

    monkeypatch.setattr(vectorstore_module, "_backend_adapter", None)
    monkeypatch.setattr(
        vectorstore_module.cfg,
        "get",
        lambda key: {
            "vectorstore.backend": "langchain_chroma",
        }[key],
    )

    class FakeBackend:
        def init(self):
            calls["count"] += 1

    monkeypatch.setattr(
        vectorstore_module,
        "_get_backend_adapter",
        lambda: FakeBackend(),
    )

    vectorstore_module.init()

    assert calls["count"] == 1


def test_retrieve_uses_langchain_chroma_backend(monkeypatch):
    from app import vectorstore as vectorstore_module

    monkeypatch.setattr(vectorstore_module, "_backend_adapter", None)
    monkeypatch.setattr(
        vectorstore_module.cfg,
        "get",
        lambda key: {
            "vectorstore.backend": "langchain_chroma",
        }[key],
    )

    class FakeDoc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    class FakeBackend:
        def retrieve(self, query, n_results):
            assert query == "cyber tech company"
            assert n_results == 3
            return [
                {
                    "record_id": "REC_001",
                    "document": "doc one",
                    "distance": 0.1235,
                },
                {
                    "record_id": "REC_002",
                    "document": "doc two",
                    "distance": 0.4568,
                },
            ]

    monkeypatch.setattr(
        vectorstore_module,
        "_get_backend_adapter",
        lambda: FakeBackend(),
    )

    results = vectorstore_module.retrieve("cyber tech company", n_results=3)

    assert results == [
        {"record_id": "REC_001", "document": "doc one", "distance": 0.1235},
        {"record_id": "REC_002", "document": "doc two", "distance": 0.4568},
    ]


def test_existing_collection_without_ready_marker_is_rebuilt(monkeypatch, tmp_path):
    from app import vectorstore_mgt as vectorstore_module

    monkeypatch.setattr(vectorstore_module, "_collection", None)
    monkeypatch.setattr(vectorstore_module, "_embedding_fn", object())
    monkeypatch.setattr(vectorstore_module, "_DOCS_DIR", tmp_path)

    (tmp_path / "REC_001.txt").write_text("doc one")

    deleted = {"name": None}
    created = {"collection": None}

    class FakeCollection:
        def __init__(self):
            self.add_calls = []

        def add(self, **kwargs):
            self.add_calls.append(kwargs)

    class FakeNamedCollection:
        name = "torch_records_huggingface"

    class FakeClient:
        def list_collections(self):
            return [FakeNamedCollection()]

        def delete_collection(self, name):
            deleted["name"] = name

        def create_collection(self, name, embedding_function, metadata):
            created["collection"] = FakeCollection()
            return created["collection"]

    monkeypatch.setattr(vectorstore_module.chromadb, "PersistentClient", lambda path: FakeClient())
    monkeypatch.setattr(vectorstore_module, "_get_collection_name", lambda: "torch_records_huggingface")
    monkeypatch.setattr(vectorstore_module, "_get_collection_marker_path", lambda: tmp_path / "marker.json")

    collection = vectorstore_module._get_collection()

    assert deleted["name"] == "torch_records_huggingface"
    assert collection is created["collection"]
    assert len(collection.add_calls) == 1


def test_existing_ready_collection_uses_marker_count_without_calling_count(monkeypatch, tmp_path):
    from app import vectorstore_mgt as vectorstore_module

    monkeypatch.setattr(vectorstore_module, "_collection", None)
    monkeypatch.setattr(vectorstore_module, "_embedding_fn", object())

    marker_path = tmp_path / "marker.json"
    marker_path.write_text('{"document_count": 42}')

    class FakeCollection:
        def count(self):
            raise AssertionError("count() should not be called for ready collections at startup")

    fake_collection = FakeCollection()

    class FakeNamedCollection:
        name = "torch_records_huggingface"

    class FakeClient:
        def list_collections(self):
            return [FakeNamedCollection()]

        def get_collection(self, name, embedding_function):
            return fake_collection

    monkeypatch.setattr(vectorstore_module.chromadb, "PersistentClient", lambda path: FakeClient())
    monkeypatch.setattr(vectorstore_module, "_get_collection_name", lambda: "torch_records_huggingface")
    monkeypatch.setattr(vectorstore_module, "_get_collection_marker_path", lambda: marker_path)

    collection = vectorstore_module._get_collection()

    assert collection is fake_collection


def test_create_collection_recovers_when_chroma_reports_already_exists(monkeypatch, tmp_path):
    from app import vectorstore_mgt as vectorstore_module

    monkeypatch.setattr(vectorstore_module, "_collection", None)
    monkeypatch.setattr(vectorstore_module, "_embedding_fn", object())
    monkeypatch.setattr(vectorstore_module, "_DOCS_DIR", tmp_path)

    (tmp_path / "REC_001.txt").write_text("doc one")

    deleted = {"name": None}
    create_calls = {"count": 0}

    class FakeCollection:
        def __init__(self):
            self.add_calls = []

        def add(self, **kwargs):
            self.add_calls.append(kwargs)

    fake_collection = FakeCollection()

    class FakeClient:
        def list_collections(self):
            return []

        def delete_collection(self, name):
            deleted["name"] = name

        def create_collection(self, name, embedding_function, metadata):
            create_calls["count"] += 1
            if create_calls["count"] == 1:
                raise RuntimeError("Collection [torch_records_huggingface] already exists")
            return fake_collection

    monkeypatch.setattr(vectorstore_module.chromadb, "PersistentClient", lambda path: FakeClient())
    monkeypatch.setattr(vectorstore_module, "_get_collection_name", lambda: "torch_records_huggingface")
    monkeypatch.setattr(vectorstore_module, "_get_collection_marker_path", lambda: tmp_path / "marker.json")

    collection = vectorstore_module._get_collection()

    assert deleted["name"] == "torch_records_huggingface"
    assert create_calls["count"] == 2
    assert collection is fake_collection


def test_populate_collection_passes_precomputed_embeddings(monkeypatch, tmp_path):
    from app import vectorstore as vectorstore_module

    monkeypatch.setattr(vectorstore_module, "_DOCS_DIR", tmp_path)
    monkeypatch.setattr(vectorstore_module, "_get_collection_name", lambda: "torch_records_huggingface")
    monkeypatch.setattr(
        vectorstore_module.cfg,
        "get",
        lambda key: {
            "vectorstore.batch_size": 10,
        }[key],
    )

    (tmp_path / "REC_001.txt").write_text("doc one")
    (tmp_path / "REC_002.txt").write_text("doc two")

    captured = {}

    class FakeEmbeddingFn:
        def __call__(self, documents):
            captured["documents"] = documents
            return [[0.1, 0.2], [0.3, 0.4]]

    class FakeCollection:
        def add(self, **kwargs):
            captured["add_kwargs"] = kwargs

    monkeypatch.setattr(vectorstore_module, "_get_embedding_fn", lambda: FakeEmbeddingFn())

    count = vectorstore_module._populate_collection(FakeCollection())

    assert count == 2
    assert captured["documents"] == ["doc one", "doc two"]
    assert captured["add_kwargs"]["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["add_kwargs"]["documents"] == ["doc one", "doc two"]


def test_get_collection_name_appends_provider_suffix(monkeypatch):
    from app import vectorstore_mgt as vectorstore_module

    monkeypatch.setattr(
        vectorstore_module.cfg,
        "get",
        lambda key: {
            "vectorstore.collection_name": "torch_records",
            "vectorstore.embedding_provider": "huggingface",
        }[key],
    )

    assert vectorstore_module._get_collection_name() == "torch_records_huggingface"


def test_get_embedding_fn_uses_onnx_provider(monkeypatch):
    from app import vectorstore_mgt as vectorstore_module

    calls = {"count": 0}

    class FakeOnnxEmbeddingFunction:
        def __init__(self):
            calls["count"] += 1

    monkeypatch.setattr(vectorstore_module, "_embedding_fn", None)
    monkeypatch.setattr(
        vectorstore_module.cfg,
        "get",
        lambda key: {
            "vectorstore.embedding_provider": "onnx",
            "embedding.model.onnx": "all-MiniLM-L6-v2",
        }[key],
    )
    monkeypatch.setattr(
        vectorstore_module.embedding_functions,
        "ONNXMiniLM_L6_V2",
        FakeOnnxEmbeddingFunction,
    )

    embedding_fn = vectorstore_module._get_embedding_fn()

    assert calls["count"] == 1
    assert isinstance(embedding_fn, FakeOnnxEmbeddingFunction)


def test_get_embedding_fn_uses_openai_provider(monkeypatch):
    from app import vectorstore_mgt as vectorstore_module

    captured = {}

    class FakeOpenAIEmbeddingFunction:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vectorstore_module, "_embedding_fn", None)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr(
        vectorstore_module.cfg,
        "get",
        lambda key: {
            "vectorstore.embedding_provider": "openai",
            "embedding.model.openai": "text-embedding-3-small",
            "vectorstore.embedding_api_base": "https://api.openai.example/v1",
            "vectorstore.openai_api_key_env_var": "OPENAI_API_KEY",
        }[key],
    )
    monkeypatch.setattr(
        vectorstore_module.embedding_functions,
        "OpenAIEmbeddingFunction",
        FakeOpenAIEmbeddingFunction,
    )

    vectorstore_module._get_embedding_fn()

    assert captured == {
        "api_key": "openai-test-key",
        "model_name": "text-embedding-3-small",
        "api_base": "https://api.openai.example/v1",
    }


def test_get_embedding_fn_uses_huggingface_provider(monkeypatch):
    from app import vectorstore_mgt as vectorstore_module

    captured = {}

    class FakeHuggingFaceEmbeddingFunction:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vectorstore_module, "_embedding_fn", None)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-test-token")
    monkeypatch.setattr(
        vectorstore_module.cfg,
        "get",
        lambda key: {
            "vectorstore.embedding_provider": "huggingface",
            "embedding.model.huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            "vectorstore.huggingface_api_key_env_var": "HUGGINGFACE_TOKEN",
        }[key],
    )
    monkeypatch.setattr(
        vectorstore_module.embedding_functions,
        "HuggingFaceEmbeddingFunction",
        FakeHuggingFaceEmbeddingFunction,
    )

    vectorstore_module._get_embedding_fn()

    assert captured == {
        "api_key": "hf-test-token",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    }


def test_get_embedding_fn_rejects_unknown_provider(monkeypatch):
    from app import vectorstore_mgt as vectorstore_module

    monkeypatch.setattr(vectorstore_module, "_embedding_fn", None)
    monkeypatch.setattr(
        vectorstore_module.cfg,
        "get",
        lambda key: {
            "vectorstore.embedding_provider": "unsupported",
        }[key],
    )

    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        vectorstore_module._get_embedding_fn()


def test_safe_huggingface_embedding_fn_raises_clear_error_on_api_error():
    from app.vectorstore import _SafeHuggingFaceEmbeddingFunction

    embedding_fn = _SafeHuggingFaceEmbeddingFunction.__new__(_SafeHuggingFaceEmbeddingFunction)
    embedding_fn.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    class FakeResponse:
        def json(self):
            return {"error": "Model is loading", "estimated_time": 12.3}

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

    embedding_fn._session = FakeSession()
    embedding_fn._api_url = "https://example.test/embed"

    with pytest.raises(RuntimeError, match="HuggingFace embedding request failed"):
        embedding_fn(["hello"])


def test_safe_huggingface_embedding_fn_uses_router_endpoint(monkeypatch):
    from app.vectorstore import _SafeHuggingFaceEmbeddingFunction

    original_init = _SafeHuggingFaceEmbeddingFunction.__mro__[1].__init__

    def fake_parent_init(self, *args, **kwargs):
        self.model_name = kwargs["model_name"]
        self._api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/old"

    monkeypatch.setattr(
        _SafeHuggingFaceEmbeddingFunction.__mro__[1],
        "__init__",
        fake_parent_init,
    )
    try:
        embedding_fn = _SafeHuggingFaceEmbeddingFunction(
            api_key="hf-test-token",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    finally:
        monkeypatch.setattr(
            _SafeHuggingFaceEmbeddingFunction.__mro__[1],
            "__init__",
            original_init,
        )

    assert (
        embedding_fn._api_url
        == "https://router.huggingface.co/hf-inference/models/"
        "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    )
