"""tile-memory — Lossy, reconstructive memory for AI systems.

Public API
----------
- :class:`Tile` — Core data structure.
- :class:`TileEncoder` — Compress content into Tiles.
- :class:`TileDecoder` — Reconstruct content from Tiles.
- :class:`TelephoneGame` — Simulate the telephone game.
- :class:`RateDistortion` — Rate-distortion analysis.
- :mod:`tile_memory.metrics` — Fact survival, novelty, lattice-snap metrics.
"""

from .decoder import DecodeResult, TileDecoder
from .encoder import TileEncoder
from .metrics import fact_survival_rate, lattice_snap_rate, novel_content_fraction
from .rate_distortion import Curve, CurvePoint, RateDistortion
from .telephone import Analysis, RoundResult, TelephoneGame
from .tile import Tile

__all__ = [
    # Core
    "Tile",
    "TileEncoder",
    "TileDecoder",
    "DecodeResult",
    # Telephone game
    "TelephoneGame",
    "RoundResult",
    "Analysis",
    # Rate-distortion
    "RateDistortion",
    "Curve",
    "CurvePoint",
    # Metrics
    "fact_survival_rate",
    "novel_content_fraction",
    "lattice_snap_rate",
]

__version__ = "0.1.0"
