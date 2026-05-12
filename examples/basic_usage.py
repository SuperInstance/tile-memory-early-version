#!/usr/bin/env python3
"""Basic usage example for the tile-memory library.

Demonstrates: encode → decode → collective decode → rate-distortion analysis.
"""

from tile_memory import (
    TileEncoder,
    TileDecoder,
    RateDistortion,
    fact_survival_rate,
    novel_content_fraction,
    lattice_snap_rate,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Encode content into a Tile
    # ------------------------------------------------------------------
    encoder = TileEncoder()

    content = (
        "Alice discovered 47 critical bugs at 3:47 AM on Tuesday. "
        "The deployment was immediately rolled back. "
        "Bob said this was 'the worst release in company history'."
    )
    tile = encoder.encode(content, salience_tags=["bugs", "Alice", "deployment"])
    print("=== Encoded Tile ===")
    print(f"  Compression ratio: {tile.compression_ratio:.1f}x")
    print(f"  Emotional valence: {tile.emotional_valence:.2f}")
    print(f"  Constraint keys:   {list(tile.constraints.keys())}")
    print(f"  Proper nouns:      {tile.constraints.get('proper_nouns', [])}")
    print(f"  Numbers:           {tile.constraints.get('numbers', [])}")
    print(f"  Quotes:            {tile.constraints.get('quotes', [])}")
    print()

    # ------------------------------------------------------------------
    # 2. Decode the Tile with context
    # ------------------------------------------------------------------
    decoder = TileDecoder()
    result = decoder.decode(tile, context="We were discussing the production incident.")
    print("=== Decoded ===")
    print(f"  Reconstruction: {result.reconstruction}")
    print(f"  Confidence:     {result.confidence:.2f}")
    print(f"  From tile:      {len(result.from_tile)} items")
    print(f"  Inferred:       {len(result.inferred)} items")
    print()

    # ------------------------------------------------------------------
    # 3. Encode a second Tile and do collective decode
    # ------------------------------------------------------------------
    content2 = (
        "The team meeting on Wednesday confirmed the rollback. "
        "Charlie estimated 3 days to fix the 47 bugs."
    )
    tile2 = encoder.encode(content2, salience_tags=["rollback", "Charlie"])

    collective = decoder.decode_collective(
        [tile, tile2],
        context="Post-incident review.",
    )
    print("=== Collective Decode ===")
    print(f"  Reconstruction: {collective.reconstruction}")
    print(f"  Confidence:     {collective.confidence:.2f}")
    print()

    # ------------------------------------------------------------------
    # 4. Rate-distortion analysis
    # ------------------------------------------------------------------
    rd = RateDistortion()
    curve = rd.compute_curve([tile, tile2], context="Post-incident review.")
    print("=== Rate-Distortion Curve ===")
    for pt in curve.points:
        print(f"  rate={pt.rate:.3f}  distortion={pt.distortion:.3f}")
    print(f"  Optimal rate (R*): {curve.optimal_rate:.3f}")
    print()

    # Context discount: how much does context help?
    discount = rd.context_discount(
        tile,
        with_context="Post-incident review.",
        without_context="",
    )
    print(f"  Context discount: {discount:.3f} (positive = context helped)")
    print()

    # Creativity score
    creativity = rd.creativity_score(content, result.reconstruction)
    print(f"  Creativity score: {creativity:.3f}")
    print()

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    # Novel content fraction
    novelty = novel_content_fraction(content, result.reconstruction)
    print(f"  Novel content fraction: {novelty:.3f}")

    # Fact survival across tiles
    facts = {"alice": "Alice", "bug_count": "47", "rollback": "rolled back"}
    survival = fact_survival_rate([tile, tile2], facts)
    print(f"  Fact survival: {survival}")

    # Lattice snap rate
    hallucinations = ["Dave", "Eve", "Wednesday"]
    valid_set = {"Alice", "Bob", "Charlie", "Wednesday", "Tuesday"}
    snap = lattice_snap_rate(hallucinations, valid_set)
    print(f"  Lattice snap rate: {snap:.3f} ({snap*100:.0f}% of hallucinations are plausible)")
    print()

    print("Done. Tiles are lossy. That's the feature.")


if __name__ == "__main__":
    main()
