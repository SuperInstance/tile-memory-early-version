"""Telephone game simulation — round-by-round lossy transmission."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from .encoder import TileEncoder
from .tile import Tile


@dataclass
class RoundResult:
    """Result of a single telephone-game round.

    Attributes:
        round_number: Which round this is (0-indexed).
        model: The model that processed this round.
        input_text: What went into this round.
        output_text: What came out.
        facts_preserved: Dict of fact-name → whether it survived this round.
        novel_claims: New claims introduced in this round (not in original).
        drift_score: How much the output drifted from the input (0.0 = identical).
    """

    round_number: int
    model: str
    input_text: str
    output_text: str
    facts_preserved: dict[str, bool] = field(default_factory=dict)
    novel_claims: list[str] = field(default_factory=list)
    drift_score: float = 0.0


@dataclass
class Analysis:
    """Analysis of a full telephone-game run.

    Attributes:
        fact_timeline: Dict mapping fact names to a list of bools (survived per round).
        drift_curve: List of drift scores, one per round.
        novel_additions_per_round: List of lists of novel claims per round.
        crystallization_round: The round at which output stopped changing significantly,
            or ``None`` if no crystallization was detected.
        summary: Human-readable summary string.
    """

    fact_timeline: dict[str, list[bool]] = field(default_factory=dict)
    drift_curve: list[float] = field(default_factory=list)
    novel_additions_per_round: list[list[str]] = field(default_factory=list)
    crystallization_round: int | None = None
    summary: str = ""


class TelephoneGame:
    """Simulate the telephone game with real API calls to LLMs.

    Sends content through successive rounds of model processing, tracking
    which facts survive and where the output crystallizes.

    Example::

        game = TelephoneGame()
        results = game.play(
            "Alice found 47 bugs at 3 AM on Tuesday.",
            rounds=6,
        )
        analysis = game.analyze(results)
        print(analysis.summary)
    """

    # Prompt template for the telephone game round
    ROUND_PROMPT = (
        "You are participating in a memory experiment. "
        "Read the following text carefully, then retell it from memory "
        "in your own words. Do NOT add any information that wasn't in the original. "
        "Keep it approximately the same length.\n\n"
        "Text to remember:\n{text}\n\n"
        "Your retelling:"
    )

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the telephone game.

        Args:
            api_key: API key for model calls. If ``None``, reads from the
                ``OPENAI_API_KEY`` environment variable (or a compatible key).
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._encoder = TileEncoder()

    def play(
        self,
        source: str,
        rounds: int = 6,
        models: list[str] | None = None,
        facts: dict[str, str] | None = None,
    ) -> list[RoundResult]:
        """Run a telephone game.

        Args:
            source: The original text to transmit.
            rounds: Number of rounds to play.
            models: List of model names to cycle through. If ``None``, uses
                ``["gpt-3.5-turbo"]`` for all rounds.
            facts: Optional dict of ``{fact_name: fact_value}`` to track
                explicitly through the chain.

        Returns:
            List of :class:`RoundResult`, one per round.
        """
        if not self._api_key:
            raise ValueError(
                "No API key provided. Pass api_key or set OPENAI_API_KEY."
            )

        models = models or ["gpt-3.5-turbo"]
        results: list[RoundResult] = []
        current_text = source

        for i in range(rounds):
            model = models[i % len(models)]
            output = self._call_model(model, current_text)

            # Track facts
            facts_preserved = (
                self._check_facts(output, facts) if facts else {}
            )

            # Detect novel claims
            novel = self._detect_novel_claims(source, output)

            # Compute drift
            drift = self._compute_drift(current_text, output)

            result = RoundResult(
                round_number=i,
                model=model,
                input_text=current_text,
                output_text=output,
                facts_preserved=facts_preserved,
                novel_claims=novel,
                drift_score=drift,
            )
            results.append(result)
            current_text = output

        return results

    def analyze(self, results: list[RoundResult]) -> Analysis:
        """Analyze a completed telephone game.

        Produces a fact-survival timeline, drift curve, novel-claim list,
        and crystallization-point detection.

        Args:
            results: List of :class:`RoundResult` from :meth:`play`.

        Returns:
            An :class:`Analysis` with full metrics.
        """
        if not results:
            return Analysis(summary="No rounds to analyse.")

        # Build fact timeline
        all_facts: set[str] = set()
        for r in results:
            all_facts.update(r.facts_preserved.keys())

        fact_timeline: dict[str, list[bool]] = {}
        for fact in sorted(all_facts):
            fact_timeline[fact] = [r.facts_preserved.get(fact, False) for r in results]

        # Drift curve
        drift_curve = [r.drift_score for r in results]

        # Novel additions
        novel_per_round = [r.novel_claims for r in results]

        # Crystallization detection: first round where drift drops below 0.1
        # for two consecutive rounds
        crystallization_round: int | None = None
        for i in range(1, len(drift_curve)):
            if drift_curve[i] < 0.1 and drift_curve[i - 1] < 0.1:
                crystallization_round = i
                break

        # Summary
        total_facts = len(all_facts)
        surviving = sum(
            1 for v in fact_timeline.values() if v and v[-1]
        ) if all_facts else 0
        max_drift = max(drift_curve) if drift_curve else 0.0

        lines = [
            f"Telephone game: {len(results)} rounds, {total_facts} tracked facts.",
            f"  Facts surviving all rounds: {surviving}/{total_facts}",
            f"  Peak drift: {max_drift:.2f}",
            f"  Total novel claims: {sum(len(n) for n in novel_per_round)}",
        ]
        if crystallization_round is not None:
            lines.append(
                f"  Crystallization detected at round {crystallization_round}."
            )
        else:
            lines.append("  No crystallization detected — content kept drifting.")

        return Analysis(
            fact_timeline=fact_timeline,
            drift_curve=drift_curve,
            novel_additions_per_round=novel_per_round,
            crystallization_round=crystallization_round,
            summary="\n".join(lines),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_model(self, model: str, text: str) -> str:
        """Call a model via OpenAI-compatible API."""
        prompt = self.ROUND_PROMPT.format(text=text)

        # Try OpenAI endpoint
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError) as exc:
            return f"[API error in round: {exc}]"

    def _check_facts(
        self, text: str, facts: dict[str, str]
    ) -> dict[str, bool]:
        """Check which facts are preserved in the text."""
        text_lower = text.lower()
        preserved: dict[str, bool] = {}
        for name, value in facts.items():
            preserved[name] = value.lower() in text_lower
        return preserved

    def _detect_novel_claims(self, original: str, output: str) -> list[str]:
        """Detect claims in output that weren't in the original."""
        orig_words = set(original.lower().split())
        out_words = set(output.lower().split())
        novel = out_words - orig_words

        # Filter to meaningful novel words (length > 4)
        return [w for w in sorted(novel) if len(w) > 4][:20]

    def _compute_drift(self, original: str, output: str) -> float:
        """Compute a simple drift score between two texts.

        Uses word overlap (Jaccard distance) as a rough drift measure.
        Returns a value in [0.0, 1.0] where 0.0 means identical.
        """
        orig = set(original.lower().split())
        out = set(output.lower().split())

        if not orig and not out:
            return 0.0
        if not orig or not out:
            return 1.0

        intersection = orig & out
        union = orig | out
        return 1.0 - (len(intersection) / len(union))
