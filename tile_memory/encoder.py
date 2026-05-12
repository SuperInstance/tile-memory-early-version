"""Tile encoder — compress raw content into lossy Tiles."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from .tile import Tile


class TileEncoder:
    """Encode raw text content into Tiles.

    The encoder uses lightweight NLP (regex, simple heuristics) to extract
    *constraint points* — the facts that a telephone game would preserve —
    and packs them into a :class:`Tile`.

    Example::

        encoder = TileEncoder()
        tile = encoder.encode(
            "Alice discovered the bug at 3:47 AM on Tuesday.",
            salience_tags=["bug", "Alice"],
        )
    """

    # Regex patterns for constraint extraction
    _NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
    _PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
    _QUOTED_RE = re.compile(r'["\u201c](.+?)["\u201d]')
    _DATE_RE = re.compile(
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        re.IGNORECASE,
    )
    _TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?:\s?[AP]M)?\b", re.IGNORECASE)
    _URL_RE = re.compile(r"https?://\S+")

    def __init__(self) -> None:
        # Simple sentiment word lists for valence estimation
        self._positive_words: set[str] = {
            "amazing", "great", "excellent", "love", "fantastic", "wonderful",
            "beautiful", "happy", "exciting", "brilliant", "perfect", "awesome",
            "discovered", "breakthrough", "success", "win", "best",
        }
        self._negative_words: set[str] = {
            "terrible", "awful", "hate", "horrible", "worst", "ugly", "sad",
            "angry", "failed", "failure", "broken", "bug", "crash", "error",
            "disaster", "problem", "issue", "critical", "dangerous",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, content: str, salience_tags: list[str] | None = None) -> Tile:
        """Compress *content* into a Tile.

        Args:
            content: Raw text to compress.
            salience_tags: Optional list of tags that mark important concepts.
                Tags present in the content boost emotional valence.

        Returns:
            A :class:`Tile` whose ``constraints`` dict holds the extracted
            constraint points.
        """
        salience_tags = [t.lower() for t in (salience_tags or [])]

        constraints = self._extract_constraints(content)
        context_required = self._infer_context_required(content, constraints)
        emotional_valence = self._compute_emotional_valence(content, salience_tags)
        compression_ratio = self._compute_compression_ratio(content, constraints)

        tile = Tile.from_content(
            content,
            constraints=constraints,
            context_required=context_required,
            emotional_valence=emotional_valence,
            compression_ratio=compression_ratio,
        )
        return tile

    def encode_batch(self, contents: list[str]) -> list[Tile]:
        """Encode a batch of content strings into Tiles.

        Args:
            contents: List of raw text strings.

        Returns:
            List of :class:`Tile` objects, one per input string.
        """
        return [self.encode(c) for c in contents]

    # ------------------------------------------------------------------
    # Internal extraction helpers
    # ------------------------------------------------------------------

    def _extract_constraints(self, content: str) -> dict[str, Any]:
        """Extract constraint points from content.

        Returns a dict with keys like ``proper_nouns``, ``numbers``,
        ``quotes``, ``dates``, ``times``, ``urls``, and ``key_phrases``.
        """
        constraints: dict[str, Any] = {}

        proper_nouns = list(set(self._PROPER_NOUN_RE.findall(content)))
        if proper_nouns:
            constraints["proper_nouns"] = proper_nouns

        numbers = self._NUMBER_RE.findall(content)
        if numbers:
            constraints["numbers"] = numbers

        quotes = self._QUOTED_RE.findall(content)
        if quotes:
            constraints["quotes"] = quotes

        dates = list(set(self._DATE_RE.findall(content)))
        if dates:
            constraints["dates"] = dates

        times = self._TIME_RE.findall(content)
        if times:
            constraints["times"] = times

        urls = self._URL_RE.findall(content)
        if urls:
            constraints["urls"] = urls

        # Key phrases: sentences that contain dramatic/specific markers
        key_phrases = self._extract_key_phrases(content)
        if key_phrases:
            constraints["key_phrases"] = key_phrases

        # Store a compressed summary (first sentence + last sentence)
        sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]
        if sentences:
            constraints["summary_anchor"] = (
                sentences[0] if len(sentences) == 1 else f"{sentences[0]} | {sentences[-1]}"
            )

        return constraints

    def _extract_key_phrases(self, content: str) -> list[str]:
        """Extract sentences containing high-specificity markers."""
        markers = {
            "discovered", "announced", "proven", "first", "only", "never",
            "always", "exactly", "precisely", "critical", "fatal", "massive",
            "tiny", "zero", "infinite", "impossible",
        }
        sentences = re.split(r"(?<=[.!?])\s+", content)
        key: list[str] = []
        for s in sentences:
            words = set(s.lower().split())
            if words & markers:
                key.append(s.strip())
        return key

    def _infer_context_required(
        self, content: str, constraints: dict[str, Any]
    ) -> list[str]:
        """Guess what context would be needed for reconstruction."""
        needed: list[str] = []

        if "proper_nouns" in constraints:
            for name in constraints["proper_nouns"]:
                needed.append(f"who/what is {name}")

        if "dates" in constraints or "times" in constraints:
            needed.append("temporal context / timeline")

        if any(w in content.lower() for w in ("code", "function", "module", "class")):
            needed.append("technical / code context")

        if any(w in content.lower() for w in ("project", "team", "sprint", "milestone")):
            needed.append("project context")

        return needed

    def _compute_emotional_valence(
        self, content: str, salience_tags: list[str]
    ) -> float:
        """Estimate emotional valence / salience on [0.0, 1.0].

        Uses simple word-counting heuristics: positive words boost, negative
        words boost (negativity is *also* salient), specificity markers boost,
        and explicit salience tags boost.
        """
        words = content.lower().split()
        n_pos = sum(1 for w in words if w in self._positive_words)
        n_neg = sum(1 for w in words if w in self._negative_words)
        n_tags = sum(1 for w in words if w in salience_tags)

        # Specificity markers
        specificity_markers = {
            "exactly", "precisely", "only", "never", "always", "zero",
            "first", "last", "critical", "fatal",
        }
        n_spec = sum(1 for w in words if w in specificity_markers)

        raw = 0.3  # baseline
        raw += 0.1 * math.log1p(n_pos)
        raw += 0.1 * math.log1p(n_neg)
        raw += 0.15 * math.log1p(n_tags)
        raw += 0.08 * math.log1p(n_spec)

        # Exclamation marks boost valence
        raw += 0.05 * min(content.count("!"), 3)

        return min(max(raw, 0.0), 1.0)

    def _compute_compression_ratio(
        self, content: str, constraints: dict[str, Any]
    ) -> float:
        """Compute compression ratio: original bytes / tile bytes."""
        import json

        original_bytes = len(content.encode())
        tile_bytes = len(json.dumps(constraints).encode())
        if tile_bytes == 0:
            return float(original_bytes) if original_bytes > 0 else 1.0
        return original_bytes / tile_bytes
