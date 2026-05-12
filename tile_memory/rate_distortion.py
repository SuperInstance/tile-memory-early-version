"""Rate-distortion analysis for Tile compression."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .decoder import TileDecoder
from .tile import Tile


@dataclass
class CurvePoint:
    """A single point on the rate-distortion curve.

    Attributes:
        rate: Compression level (bits / bytes of tile storage).
        distortion: Reconstruction error (0.0 = perfect, 1.0 = total loss).
    """

    rate: float
    distortion: float


@dataclass
class Curve:
    """A rate-distortion curve with optimal point.

    Attributes:
        points: Ordered list of :class:`CurvePoint`.
        optimal_rate: The rate ``R*`` that minimises rate + λ·distortion.
        optimal_point: The :class:`CurvePoint` at ``R*``.
    """

    points: list[CurvePoint] = field(default_factory=list)
    optimal_rate: float = 0.0
    optimal_point: CurvePoint | None = None


class RateDistortion:
    """Rate-distortion analysis for Tiles.

    Measures the trade-off between compression (rate) and reconstruction
    fidelity (distortion), following the Tile Compression Theorem.

    Example::

        rd = RateDistortion()
        curve = rd.compute_curve(tiles)
        print(f"Optimal rate: {curve.optimal_rate:.2f}")
    """

    def __init__(self, decoder: TileDecoder | None = None) -> None:
        self._decoder = decoder or TileDecoder()

    def compute_curve(
        self,
        tiles: list[Tile],
        context: str = "",
        lambda_weight: float = 1.0,
    ) -> Curve:
        """Compute a rate-distortion curve for a set of Tiles.

        Args:
            tiles: List of Tiles to analyse.
            context: Optional context for reconstruction.
            lambda_weight: Trade-off weight for the Lagrangian ``R + λ·D``.

        Returns:
            A :class:`Curve` with points and the optimal rate ``R*``.
        """
        if not tiles:
            return Curve()

        points: list[CurvePoint] = []

        for tile in tiles:
            rate = 1.0 / tile.compression_ratio if tile.compression_ratio > 0 else 0.0
            distortion = self._estimate_distortion(tile, context)
            points.append(CurvePoint(rate=rate, distortion=distortion))

        # Sort by rate
        points.sort(key=lambda p: p.rate)

        # Find optimal point: minimise rate + lambda * distortion
        best_score = float("inf")
        optimal_point: CurvePoint | None = None

        for pt in points:
            score = pt.rate + lambda_weight * pt.distortion
            if score < best_score:
                best_score = score
                optimal_point = pt

        optimal_rate = optimal_point.rate if optimal_point else 0.0

        return Curve(
            points=points,
            optimal_rate=optimal_rate,
            optimal_point=optimal_point,
        )

    def context_discount(
        self,
        tile: Tile,
        with_context: str,
        without_context: str = "",
    ) -> float:
        """Measure how much context reduces reconstruction distortion.

        Args:
            tile: The Tile to reconstruct.
            with_context: Context string to supply during reconstruction.
            without_context: Baseline context (default: empty).

        Returns:
            A discount value: ``distortion_without - distortion_with``.
            Positive means context helped.
        """
        d_with = self._estimate_distortion(tile, with_context)
        d_without = self._estimate_distortion(tile, without_context)
        return d_without - d_with

    def creativity_score(
        self,
        original: str,
        reconstruction: str,
    ) -> float:
        """Measure the fraction of novel but plausible content.

        Counts words in the reconstruction that don't appear in the original,
        weighted by plausibility (word length and common-pattern heuristics).

        Args:
            original: The source text.
            reconstruction: The reconstructed text.

        Returns:
            A creativity score in ``[0.0, 1.0]``.
        """
        orig_words = set(original.lower().split())
        recon_words = set(reconstruction.lower().split())

        if not recon_words:
            return 0.0

        novel = recon_words - orig_words
        if not novel:
            return 0.0

        # Plausibility: words of reasonable length (3-12 chars) are plausible
        plausible = [w for w in novel if 3 <= len(w) <= 12]
        if not plausible:
            return 0.0

        return len(plausible) / len(recon_words)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_distortion(self, tile: Tile, context: str) -> float:
        """Estimate reconstruction distortion for a Tile.

        Uses heuristic: distortion inversely proportional to constraint count
        and emotional valence, boosted by context availability.
        """
        n_constraints = sum(
            len(v) if isinstance(v, list) else 1
            for v in tile.constraints.values()
        )

        # More constraints → lower distortion
        constraint_factor = 1.0 / (1.0 + math.log1p(n_constraints))

        # Higher valence → lower distortion (emotionally salient = better remembered)
        valence_factor = 1.0 - (0.3 * tile.emotional_valence)

        # Context helps
        context_factor = 0.7 if context else 1.0

        distortion = constraint_factor * valence_factor * context_factor
        return min(max(distortion, 0.0), 1.0)
