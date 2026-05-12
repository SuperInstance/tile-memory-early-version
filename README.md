# tile-memory

**Lossy, reconstructive memory for AI systems.**

Based on the Tile Compression Theorem: *forgetting is the feature, not the bug*.

## Why?

AI systems that remember everything bloat. AI systems that forget everything lose. The Tile Compression Theorem says: compress memories into **Tiles** — constraint points that survive the telephone game — and reconstruct on demand with fresh context. The reconstruction won't be exact. That's the point. It'll be *good enough*, and sometimes *creatively better*.

## What's a Tile?

A Tile stores the immortal facts — proper nouns, numbers, dramatic phrases — and throws away the prose. When you need the memory back, the decoder uses the Tile's constraints + fresh context to reconstruct. Like how you remember your friend's wedding: you don't recall the exact words, but you remember the key facts and can retell it.

## Quick Start

```bash
pip install -e .
```

```python
from tile_memory import TileEncoder, TileDecoder, TelephoneGame

# Encode
encoder = TileEncoder()
tile = encoder.encode(
    "Alice discovered 47 critical bugs at 3:47 AM on Tuesday. "
    "The deployment was immediately rolled back.",
    salience_tags=["bugs", "Alice", "deployment"],
)

print(f"Compression: {tile.compression_ratio:.1f}x")
print(f"Constraints: {list(tile.constraints.keys())}")
print(f"Valence: {tile.emotional_valence:.2f}")

# Decode
decoder = TileDecoder()
result = decoder.decode(tile, context="We were discussing the production incident.")
print(f"Reconstruction: {result.reconstruction}")
print(f"Confidence: {result.confidence:.2f}")

# Telephone game (requires API key)
# game = TelephoneGame(api_key="your-key")
# results = game.play(
#     "Alice discovered 47 critical bugs at 3:47 AM on Tuesday.",
#     rounds=6,
#     facts={"alice": "Alice", "bug_count": "47", "time": "3:47 AM"},
# )
# analysis = game.analyze(results)
# print(analysis.summary)
```

## Architecture

```
Content → TileEncoder → Tile (constraints only)
Tile + Context → TileDecoder → Reconstructed content

TelephoneGame: Content → Round 1 → Round 2 → ... → Round N
               Track: fact survival, drift, crystallization point

RateDistortion: Measure compression vs. fidelity trade-off
```

## Metrics

- **Fact survival rate** — Which facts make it through N rounds
- **Novel content fraction** — How much reconstruction is new (not wrong, just new)
- **Lattice snap rate** — What fraction of "hallucinations" are structurally plausible

## The Theory

The Tile Compression Theorem models memory as lossy compression with a rate-distortion curve. There's an optimal rate R* where you store *just enough* constraints to reconstruct well with context. Store more → wasted space. Store less → unrecoverable loss.

The telephone game is the experimental proof: facts that survive 6+ rounds of retelling are your Tiles. Everything else is reconstructable.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
