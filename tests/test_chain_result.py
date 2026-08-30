from types import SimpleNamespace

from rag import chain
from rag.chain import _MODEL, _GeneratedAnswer, _public_sources
from rag.vectorstore import RetrievedChunk


def _chunk(chunk_id: str, title: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document="document",
        artist="Damso",
        title=title,
        album=None,
        year=None,
        chunk_index=0,
        distance=0.2,
    )


def test_expected_groq_model_is_configured():
    assert _MODEL == "openai/gpt-oss-120b"


def test_groq_call_uses_low_reasoning_and_json_mode(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content='{"status":"insufficient"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(chain, "_get_groq_client", lambda: fake_client)

    chain._call_groq(
        [{"role": "user", "content": "question"}],
        max_completion_tokens=100,
        json_mode=True,
    )

    assert captured["model"] == "openai/gpt-oss-120b"
    assert captured["reasoning_effort"] == "low"
    assert captured["include_reasoning"] is False
    assert captured["response_format"] == {"type": "json_object"}


def test_public_sources_follow_model_citations_and_deduplicate_songs():
    chunks = [
        _chunk("damso:song-0001:chunk-000", "Amnésie"),
        _chunk("damso:song-0001:chunk-001", "Amnésie"),
        _chunk("damso:song-0002:chunk-000", "Mort"),
    ]

    sources = _public_sources(chunks, [1, 2, 3], "answered")

    assert [source.title for source in sources] == ["Amnésie", "Mort"]
    assert sources[0].id == "damso:song-0001"


def test_insufficient_answer_exposes_no_sources():
    chunks = [_chunk("damso:song-0001:chunk-000", "Amnésie")]
    generated = _GeneratedAnswer(
        status="insufficient",
        answer="Le corpus ne permet pas de répondre.",
        cited_sources=[],
    )

    assert _public_sources(chunks, generated.cited_sources, generated.status) == []


def test_answer_with_no_valid_citations_exposes_no_fallback_sources():
    chunks = [_chunk("damso:song-0001:chunk-000", "Amnésie")]

    assert _public_sources(chunks, [], "answered") == []
    assert _public_sources(chunks, [99], "answered") == []
