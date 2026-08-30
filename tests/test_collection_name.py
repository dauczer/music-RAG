"""Tests for _collection_name slug normalisation."""

from rag.vectorstore import _collection_name


def test_spaces_replaced():
    assert _collection_name("MC Solaar") == "mc_solaar"


def test_diacritics_stripped():
    # "Suprême NTM" → strip accent on è → "supreme_ntm"
    assert _collection_name("Suprême NTM") == "supreme_ntm"


def test_already_ascii_lowercase():
    assert _collection_name("damso") == "damso"
