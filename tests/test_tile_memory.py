"""Tests for tile-memory."""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/..")

from tile_memory import (
    Tile, TileEncoder, TileDecoder, DecodeResult,
    TelephoneGame, RateDistortion, Curve, CurvePoint,
    fact_survival_rate, novel_content_fraction, lattice_snap_rate,
)


# ─── Tile ──────────────────────────────────────────────────────────────────────

def test_tile_creation():
    t = Tile()
    assert t.id  # UUID generated
    assert t.compression_ratio == 1.0
    assert t.emotional_valence == 0.5
    assert t.access_count == 0


def test_tile_from_content():
    t = Tile.from_content("Hello world", constraints={"key": "val"})
    assert t.source_hash  # SHA-256
    assert t.constraints == {"key": "val"}
    assert len(t.source_hash) == 64


def test_tile_from_content_hash_deterministic():
    t1 = Tile.from_content("same content")
    t2 = Tile.from_content("same content")
    assert t1.source_hash == t2.source_hash


def test_tile_from_content_hash_differs():
    t1 = Tile.from_content("content A")
    t2 = Tile.from_content("content B")
    assert t1.source_hash != t2.source_hash


def test_tile_touch():
    t = Tile()
    assert t.access_count == 0
    t.touch()
    assert t.access_count == 1
    t.touch()
    assert t.access_count == 2


def test_tile_size_estimate():
    t = Tile(constraints={"a": "hello", "b": [1, 2, 3]})
    size = t.size_estimate()
    assert size > 0


# ─── Encoder ───────────────────────────────────────────────────────────────────

def test_encode_basic():
    enc = TileEncoder()
    tile = enc.encode("Alice discovered the bug at 3:47 AM on Tuesday.")
    assert isinstance(tile, Tile)
    assert "proper_nouns" in tile.constraints
    assert "Alice" in tile.constraints["proper_nouns"]
    assert "numbers" in tile.constraints


def test_encode_extracts_proper_nouns():
    enc = TileEncoder()
    tile = enc.encode("Bob and Charlie went to Paris.")
    assert "Bob" in tile.constraints["proper_nouns"]
    assert "Charlie" in tile.constraints["proper_nouns"]


def test_encode_extracts_numbers():
    enc = TileEncoder()
    tile = enc.encode("There are 42 bugs and 7 features.")
    nums = tile.constraints["numbers"]
    assert "42" in nums
    assert "7" in nums


def test_encode_extracts_quotes():
    enc = TileEncoder()
    tile = enc.encode('She said "hello world" to everyone.')
    assert "hello world" in tile.constraints["quotes"]


def test_encode_emotional_valence_positive():
    enc = TileEncoder()
    tile = enc.encode("This is amazing and brilliant!")
    assert tile.emotional_valence > 0.4  # boosted by positive words


def test_encode_emotional_valence_neutral():
    enc = TileEncoder()
    tile = enc.encode("The data was recorded.")
    assert tile.emotional_valence <= 0.6


def test_encode_compression_ratio():
    enc = TileEncoder()
    long_text = "Alice went to the store. " * 50
    tile = enc.encode(long_text)
    assert tile.compression_ratio > 1.0


def test_encode_batch():
    enc = TileEncoder()
    tiles = enc.encode_batch(["hello", "world", "test"])
    assert len(tiles) == 3
    assert all(isinstance(t, Tile) for t in tiles)


def test_encode_with_salience_tags():
    enc = TileEncoder()
    tile = enc.encode("The bug was discovered.", salience_tags=["bug"])
    assert tile.emotional_valence > 0.3


def test_encode_context_required():
    enc = TileEncoder()
    tile = enc.encode("Alice worked on the sprint milestone.")
    assert len(tile.context_required) > 0


# ─── Decoder ───────────────────────────────────────────────────────────────────

def test_decode_basic():
    enc = TileEncoder()
    dec = TileDecoder()
    tile = enc.encode("Alice discovered the bug at 3:47 AM on Tuesday.")
    result = dec.decode(tile)
    assert isinstance(result, DecodeResult)
    assert result.reconstruction
    assert result.confidence > 0.0


def test_decode_with_context():
    enc = TileEncoder()
    dec = TileDecoder()
    tile = enc.encode("Alice discovered the bug.")
    result = dec.decode(tile, context="We were talking about the team standup.")
    assert "context" in result.reconstruction.lower() or result.confidence > 0


def test_decode_touches_tile():
    enc = TileEncoder()
    dec = TileDecoder()
    tile = enc.encode("test content")
    assert tile.access_count == 0
    dec.decode(tile)
    assert tile.access_count == 1


def test_decode_collective():
    enc = TileEncoder()
    dec = TileDecoder()
    tiles = [
        enc.encode("Alice found 47 bugs at 3 AM."),
        enc.encode("Bob reviewed the code on Tuesday."),
    ]
    result = dec.decode_collective(tiles)
    assert result.reconstruction
    assert result.confidence > 0


def test_decode_collective_empty():
    dec = TileDecoder()
    result = dec.decode_collective([])
    assert result.confidence == 0.0


# ─── Round-trip ────────────────────────────────────────────────────────────────

def test_encode_decode_roundtrip():
    enc = TileEncoder()
    dec = TileDecoder()
    original = "Alice discovered the critical bug at exactly 3:47 AM on Tuesday."
    tile = enc.encode(original)
    result = dec.decode(tile)
    # Should mention Alice
    assert "Alice" in result.reconstruction or "alice" in result.reconstruction.lower()


# ─── Metrics ───────────────────────────────────────────────────────────────────

def test_fact_survival_rate():
    enc = TileEncoder()
    tiles = [
        enc.encode("Alice found 47 bugs."),
        enc.encode("Bob reviewed the code."),
    ]
    facts = {"person": "alice"}
    timeline = fact_survival_rate(tiles, facts)
    assert "person" in timeline
    assert len(timeline["person"]) == 2


def test_novel_content_fraction():
    frac = novel_content_fraction("hello world", "hello world")
    assert frac == 0.0

    frac2 = novel_content_fraction("hello world", "hello universe")
    assert frac2 > 0.0


def test_novel_content_empty():
    assert novel_content_fraction("hello", "") == 0.0


def test_lattice_snap_rate():
    hallucinations = ["python", "rust", "brainfuck"]
    valid = {"python", "rust", "c++", "java"}
    rate = lattice_snap_rate(hallucinations, valid)
    assert rate == pytest.approx(2/3, abs=0.01)


def test_lattice_snap_rate_empty():
    assert lattice_snap_rate([], {"a", "b"}) == 0.0


# ─── Rate Distortion ──────────────────────────────────────────────────────────

def test_rate_distortion_curve():
    enc = TileEncoder()
    tiles = [
        enc.encode("Short."),
        enc.encode("Alice found 47 bugs at 3 AM on Tuesday in the critical module."),
    ]
    rd = RateDistortion()
    curve = rd.compute_curve(tiles)
    assert isinstance(curve, Curve)
    assert len(curve.points) == 2
    assert curve.optimal_point is not None


def test_rate_distortion_empty():
    rd = RateDistortion()
    curve = rd.compute_curve([])
    assert len(curve.points) == 0


def test_context_discount():
    enc = TileEncoder()
    tile = enc.encode("Alice found the bug.")
    rd = RateDistortion()
    discount = rd.context_discount(tile, with_context="debugging session")
    assert isinstance(discount, float)


def test_creativity_score():
    rd = RateDistortion()
    score = rd.creativity_score("hello world", "hello universe planet")
    assert score >= 0.0


# ─── TelephoneGame (non-API tests) ────────────────────────────────────────────

def test_telephone_game_no_key():
    game = TelephoneGame()
    with pytest.raises(ValueError, match="No API key"):
        game.play("test", rounds=1)


def test_telephone_game_analyze_empty():
    game = TelephoneGame()
    analysis = game.analyze([])
    assert "No rounds" in analysis.summary


def test_telephone_game_analyze_with_results():
    from tile_memory.telephone import RoundResult
    results = [
        RoundResult(0, "test", "hello", "hello", {"a": True}, [], 0.0),
        RoundResult(1, "test", "hello", "world", {"a": False}, ["world"], 0.5),
    ]
    game = TelephoneGame()
    analysis = game.analyze(results)
    assert "a" in analysis.fact_timeline
    assert len(analysis.drift_curve) == 2


# ─── Import ────────────────────────────────────────────────────────────────────

def test_version():
    import tile_memory
    assert hasattr(tile_memory, "__version__")
