from rag import vectorstore


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return [[3.0, 4.0]]


class _Array:
    def tolist(self):
        return [[0.6, 0.8]]


class _LocalModel:
    def __init__(self):
        self.inputs = []
        self.options = {}

    def encode(self, inputs, **options):
        self.inputs = inputs
        self.options = options
        return _Array()


def test_remote_query_embedding_uses_e5_prefix_and_normalizes(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(vectorstore.requests, "post", post)

    embeddings = vectorstore._embed_queries(["mémoire et rupture"])

    assert "intfloat/multilingual-e5-small" in captured["url"]
    assert captured["json"] == {"inputs": ["query: mémoire et rupture"]}
    assert embeddings == [[0.6, 0.8]]


def test_corpus_embedding_is_local_and_uses_passage_prefix(monkeypatch):
    model = _LocalModel()
    monkeypatch.setattr(vectorstore, "_get_local_embedding_model", lambda: model)

    embeddings = vectorstore._embed_passages_locally(["Artist: Damso\nLyrics excerpt: texte"])

    assert model.inputs == ["passage: Artist: Damso\nLyrics excerpt: texte"]
    assert model.options["normalize_embeddings"] is True
    assert embeddings == [[0.6, 0.8]]
