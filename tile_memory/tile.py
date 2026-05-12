"""Core Tile data structure for the Tile Compression Theorem."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class Tile:
    """A lossy, reconstructive memory unit.

    A Tile captures the *constraint points* of some content — the facts that
    survive every round of a telephone game — while discarding everything else.
    When combined with fresh context during decoding, the Tile allows
    reconstructive (often creative) recall.

    Attributes:
        id: Unique identifier (UUID4).
        source_hash: SHA-256 hash of the original content, used for tracking.
        compression_ratio: ``original_bytes / tile_bytes`` — higher means more
            compression (more forgetting).
        constraints: The immortal facts — constraint points that survive all
            reconstruction rounds.  Stored as a plain dict so any JSON-serialisable
            structure works.
        context_required: Hints about what external context is needed for
            successful reconstruction (e.g. ``["who is Alice", "project timeline"]``).
        emotional_valence: Salience weight in ``[0.0, 1.0]``.  Higher values mean
            the content is more emotionally charged or important.
        created_at: When the tile was encoded.
        accessed_at: When the tile was last decoded.
        access_count: How many times the tile has been decoded.
        round_number: How many telephone-game rounds this tile has survived
            without additional encoding.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_hash: str = ""
    compression_ratio: float = 1.0
    constraints: dict[str, Any] = field(default_factory=dict)
    context_required: list[str] = field(default_factory=list)
    emotional_valence: float = 0.5
    created_at: datetime = field(default_factory=_now)
    accessed_at: datetime = field(default_factory=_now)
    access_count: int = 0
    round_number: int = 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_content(
        cls,
        content: str,
        constraints: dict[str, Any] | None = None,
        context_required: list[str] | None = None,
        emotional_valence: float = 0.5,
        compression_ratio: float = 1.0,
        round_number: int = 0,
    ) -> Tile:
        """Create a Tile from raw content, automatically hashing the source."""
        return cls(
            source_hash=hashlib.sha256(content.encode()).hexdigest(),
            compression_ratio=compression_ratio,
            constraints=constraints or {},
            context_required=context_required or [],
            emotional_valence=emotional_valence,
            round_number=round_number,
        )

    def touch(self) -> None:
        """Mark the tile as accessed (bump access count and timestamp)."""
        self.accessed_at = _now()
        self.access_count += 1

    def size_estimate(self) -> int:
        """Rough byte-size estimate of the stored constraint payload."""
        import json

        return len(json.dumps(self.constraints).encode())

    def __repr__(self) -> str:  # pragma: no cover
        keys = list(self.constraints.keys())[:5]
        return (
            f"Tile(id={self.id[:8]}…, "
            f"constraints={len(self.constraints)}, "
            f"ratio={self.compression_ratio:.1f}x, "
            f"valence={self.emotional_valence:.2f}, "
            f"round={self.round_number}, "
            f"keys={keys})"
        )
