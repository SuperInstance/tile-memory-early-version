"""Metrics for the Tile Compression Theorem."""

from __future__ import annotations

from typing import Any

from .tile import Tile


def fact_survival_rate(
    tiles: list[Tile],
    facts: dict[str, str],
) -> dict[str, list[bool]]:
    """Check which facts survive across a chain of Tiles.

    Args:
        tiles: Ordered list of Tiles (e.g. from successive telephone rounds).
        facts: Dict of ``{fact_name: fact_value}`` to check.

    Returns:
        Dict mapping each fact name to a list of bools — one per Tile — where
        ``True`` means the fact was found in that Tile's constraints.
    """
    timeline: dict[str, list[bool]] = {}

    for fact_name, fact_value in facts.items():
        survival: list[bool] = []
        value_lower = fact_value.lower()

        for tile in tiles:
            # Search all constraint values for the fact
            found = False
            for v in tile.constraints.values():
                if isinstance(v, str):
                    if value_lower in v.lower():
                        found = True
                        break
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and value_lower in item.lower():
                            found = True
                            break
                    if found:
                        break
            survival.append(found)

        timeline[fact_name] = survival

    return timeline


def novel_content_fraction(original: str, reconstruction: str) -> float:
    """Compute the fraction of the reconstruction that is novel content.

    Novel content = words in the reconstruction that don't appear in the
    original.

    Args:
        original: The source text.
        reconstruction: The reconstructed text.

    Returns:
        Fraction in ``[0.0, 1.0]``.
    """
    orig_words = set(original.lower().split())
    recon_words = reconstruction.lower().split()

    if not recon_words:
        return 0.0

    novel = sum(1 for w in recon_words if w not in orig_words)
    return novel / len(recon_words)


def lattice_snap_rate(
    hallucinations: list[str],
    valid_set: set[str],
) -> float:
    """Compute what fraction of "hallucinations" are structurally plausible.

    A hallucination *snaps to the lattice* if it matches a known valid item
    from ``valid_set``.  This measures whether the model's fabrications are
    at least in the right neighbourhood of the solution space.

    Args:
        hallucinations: List of claimed/hallucinated items.
        valid_set: Set of known valid items.

    Returns:
        Fraction of hallucinations that are in ``valid_set``.
    """
    if not hallucinations:
        return 0.0

    valid_lower = {v.lower() for v in valid_set}
    snaps = sum(1 for h in hallucinations if h.lower() in valid_lower)
    return snaps / len(hallucinations)
