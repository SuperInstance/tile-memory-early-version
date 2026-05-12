"""Tile decoder — reconstruct content from Tiles and context."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .tile import Tile


@dataclass
class DecodeResult:
    """Result of a single tile decoding operation.

    Attributes:
        reconstruction: The reconstructed text.
        inferred: Items that were inferred (not directly in the tile).
        from_tile: Items pulled directly from tile constraints.
        confidence: Overall confidence score in [0.0, 1.0].
    """

    reconstruction: str
    inferred: list[str] = field(default_factory=list)
    from_tile: list[str] = field(default_factory=list)
    confidence: float = 1.0


class TileDecoder:
    """Decode Tiles back into (reconstructed) text.

    The decoder uses a Tile's constraint points as anchors and fills in the
    gaps using any supplied context.  Because Tiles are lossy, reconstruction
    is *reconstructive* — it may introduce novel but plausible content.

    Example::

        decoder = TileDecoder()
        result = decoder.decode(tile, context="We were talking about the team standup.")
        print(result.reconstruction)
    """

    def decode(self, tile: Tile, context: str = "") -> DecodeResult:
        """Reconstruct content from a single Tile.

        Args:
            tile: The Tile to decode.
            context: Optional context string to aid reconstruction.

        Returns:
            A :class:`DecodeResult` with the reconstructed text and metadata.
        """
        tile.touch()

        constraints = tile.constraints
        inferred: list[str] = []
        from_tile: list[str] = []
        parts: list[str] = []

        # Summary anchor is the backbone
        anchor = constraints.get("summary_anchor", "")
        if anchor:
            from_tile.append(f"anchor: {anchor}")
            parts.append(anchor)

        # Reconstruct proper nouns
        proper_nouns = constraints.get("proper_nouns", [])
        if proper_nouns:
            from_tile.append(f"entities: {', '.join(proper_nouns)}")

        # Reconstruct numbers/facts
        numbers = constraints.get("numbers", [])
        if numbers:
            from_tile.append(f"numbers: {', '.join(numbers)}")

        # Key phrases carry dramatic content
        key_phrases = constraints.get("key_phrases", [])
        for phrase in key_phrases:
            from_tile.append(f"key: {phrase}")
            if phrase not in parts:
                parts.append(phrase)

        # Quotes
        quotes = constraints.get("quotes", [])
        for q in quotes:
            from_tile.append(f"quote: {q}")
            parts.append(f'"{q}"')

        # Temporal anchors
        dates = constraints.get("dates", [])
        times = constraints.get("times", [])
        if dates or times:
            temporal = " and ".join(
                filter(None, [", ".join(dates), ", ".join(times)])
            )
            from_tile.append(f"temporal: {temporal}")

        # Use context to fill gaps
        if context:
            context_nouns = self._extract_context_entities(context)
            for noun in context_nouns:
                if noun not in proper_nouns:
                    inferred.append(f"from context: {noun}")

        # Build reconstruction
        if not parts and anchor:
            reconstruction = self._prose_from_constraints(constraints, context)
        elif parts:
            reconstruction = self._assemble_reconstruction(parts, constraints, context)
        else:
            reconstruction = context if context else "[insufficient constraints for reconstruction]"

        # Compute confidence
        n_constraints = sum(len(v) if isinstance(v, list) else 1 for v in constraints.values())
        context_boost = 0.15 if context else 0.0
        confidence = min(0.3 + 0.1 * n_constraints + context_boost, 1.0)

        return DecodeResult(
            reconstruction=reconstruction,
            inferred=inferred,
            from_tile=from_tile,
            confidence=confidence,
        )

    def decode_collective(
        self, tiles: list[Tile], context: str = ""
    ) -> DecodeResult:
        """Reconstruct content from multiple Tiles.

        Merges constraint points from all tiles, votes on conflicting facts,
        and returns a unified reconstruction with confidence scores.

        Args:
            tiles: List of Tiles to decode collectively.
            context: Optional context string.

        Returns:
            A :class:`DecodeResult` combining all tiles.
        """
        if not tiles:
            return DecodeResult(reconstruction="[no tiles]", confidence=0.0)

        for t in tiles:
            t.touch()

        # Merge constraints across tiles
        merged = self._merge_constraints(tiles)
        inferred: list[str] = []
        from_tile: list[str] = []

        # Build from merged constraints
        all_nouns = merged.get("proper_nouns", [])
        all_numbers = merged.get("numbers", [])
        all_phrases = merged.get("key_phrases", [])
        all_quotes = merged.get("quotes", [])
        anchor = merged.get("summary_anchor", "")

        parts: list[str] = []
        if anchor:
            from_tile.append(f"anchor: {anchor}")
            parts.append(anchor)
        for phrase in all_phrases:
            from_tile.append(f"key: {phrase}")
            if phrase not in parts:
                parts.append(phrase)
        for q in all_quotes:
            from_tile.append(f"quote: {q}")
            parts.append(f'"{q}"')

        reconstruction = self._assemble_reconstruction(parts, merged, context)

        # Confidence: more tiles and more context → higher
        n_constraints = sum(
            len(v) if isinstance(v, list) else 1 for v in merged.values()
        )
        confidence = min(
            0.4 + 0.05 * len(tiles) + 0.1 * n_constraints + (0.1 if context else 0.0),
            1.0,
        )

        return DecodeResult(
            reconstruction=reconstruction,
            inferred=inferred,
            from_tile=from_tile,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_context_entities(self, context: str) -> list[str]:
        """Extract proper nouns from context string."""
        return list(set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", context)))

    def _merge_constraints(self, tiles: list[Tile]) -> dict[str, Any]:
        """Merge constraints from multiple tiles with voting."""
        merged: dict[str, Any] = {}

        # Collect all values for list-type constraints
        list_keys = {"proper_nouns", "numbers", "dates", "times", "urls", "key_phrases", "quotes"}
        counters: dict[str, Counter] = {k: Counter() for k in list_keys}

        anchors: list[str] = []

        for tile in tiles:
            for key in list_keys:
                vals = tile.constraints.get(key, [])
                if isinstance(vals, list):
                    for v in vals:
                        counters[key][v] += 1

            a = tile.constraints.get("summary_anchor", "")
            if a:
                anchors.append(a)

        for key, counter in counters.items():
            # Keep items that appear in at least one tile
            merged[key] = list(counter.keys())

        # Use the most common anchor
        if anchors:
            anchor_counts = Counter(anchors)
            merged["summary_anchor"] = anchor_counts.most_common(1)[0][0]

        return merged

    def _prose_from_constraints(
        self, constraints: dict[str, Any], context: str
    ) -> str:
        """Generate prose when only constraints are available (no key phrases)."""
        parts: list[str] = []

        nouns = constraints.get("proper_nouns", [])
        numbers = constraints.get("numbers", [])
        anchor = constraints.get("summary_anchor", "")

        if anchor:
            parts.append(anchor)
        elif nouns:
            parts.append(f"Regarding {' '.join(nouns[:3])}")

        if numbers:
            parts.append(f"involving {', '.join(numbers[:5])}")

        if context:
            parts.append(context)

        return ". ".join(parts) + "." if parts else "[no content]"

    def _assemble_reconstruction(
        self,
        parts: list[str],
        constraints: dict[str, Any],
        context: str,
    ) -> str:
        """Assemble a reconstruction from parts, constraints, and context."""
        base = ". ".join(parts)

        # Append temporal context if available
        dates = constraints.get("dates", [])
        times = constraints.get("times", [])
        temporal_parts = []
        if dates:
            temporal_parts.append(f"on {'/'.join(dates)}")
        if times:
            temporal_parts.append(f"at {'/'.join(times)}")
        if temporal_parts:
            base += " (" + ", ".join(temporal_parts) + ")"

        # Append context if provided
        if context and context.strip():
            base += f" [context: {context.strip()}]"

        # Ensure ends with period
        if not base.endswith("."):
            base += "."

        return base
