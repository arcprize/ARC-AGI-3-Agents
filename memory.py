"""Memory module for storing and retrieving program execution traces and hypotheses."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""

    key: str
    value: Any
    tags: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    hits: int = 0


class TraceMemory:
    """Persistent memory for storing program execution traces and hypotheses."""

    def __init__(self, path: str = "pmll_memory.jsonl") -> None:
        """Initialize the trace memory.

        Args:
            path: Path to the JSON lines file for persistent storage.
        """
        self._path = Path(path)
        self._store: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load memory entries from disk."""
        if not self._path.exists():
            return
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    e = MemoryEntry(**d)
                    self._store[e.key] = e
                except Exception:
                    pass

    def _flush(self) -> None:
        """Flush memory entries to disk."""
        with self._path.open("w") as f:
            for e in self._store.values():
                f.write(json.dumps(e.__dict__) + "\n")

    def set_memory(
        self, key: str, value: Any, tags: Optional[list[str]] = None
    ) -> None:
        """Set a memory entry.

        Args:
            key: Unique key for the memory entry.
            value: Value to store (must be JSON-serializable).
            tags: Optional list of tags for filtering.
        """
        existing = self._store.get(key)
        self._store[key] = MemoryEntry(
            key=key,
            value=value,
            tags=tags or (existing.tags if existing else []),
            ts=time.time(),
            hits=existing.hits if existing else 0,
        )
        self._flush()

    def get_memory(self, key: str) -> Optional[Any]:
        """Get a memory entry by key.

        Args:
            key: Key to look up.

        Returns:
            The value associated with the key, or None if not found.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        entry.hits += 1
        return entry.value

    def delete_memory(self, key: str) -> bool:
        """Delete a memory entry.

        Args:
            key: Key to delete.

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        existed = key in self._store
        self._store.pop(key, None)
        self._flush()
        return existed

    def list_memories(self, tag: Optional[str] = None) -> list[dict[str, Any]]:
        """List all memory entries, optionally filtered by tag.

        Args:
            tag: Optional tag to filter by.

        Returns:
            List of memory entry dictionaries.
        """
        entries = list(self._store.values())
        if tag:
            entries = [e for e in entries if tag in e.tags]
        return [e.__dict__ for e in entries]

    def program_fingerprint(self, program_tokens: list[str]) -> str:
        """Generate a fingerprint for a program.

        Args:
            program_tokens: List of program tokens.

        Returns:
            A 16-character hex fingerprint.
        """
        payload = json.dumps(sorted(program_tokens)).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def mark_failed_program(self, program_tokens: list[str], error: str) -> None:
        """Mark a program as failed with an error message.

        Args:
            program_tokens: List of program tokens.
            error: Error message (truncated to 120 characters).
        """
        fp = self.program_fingerprint(program_tokens)
        key = f"failed_program:{fp}"
        prev = self.get_memory(key) or {"count": 0, "errors": []}
        prev["count"] += 1
        prev["errors"] = (prev["errors"] + [error[:120]])[-5:]
        self.set_memory(key, prev, tags=["failed_program"])

    def is_program_hopeless(
        self, program_tokens: list[str], threshold: int = 3
    ) -> bool:
        """Check if a program has failed too many times.

        Args:
            program_tokens: List of program tokens.
            threshold: Failure count threshold.

        Returns:
            True if the program has failed at least threshold times.
        """
        fp = self.program_fingerprint(program_tokens)
        entry = self.get_memory(f"failed_program:{fp}")
        return bool(entry and entry["count"] >= threshold)

    def record_trajectory(
        self, game_id: str, step: int, summary: dict[str, Any]
    ) -> None:
        """Record a trajectory step.

        Args:
            game_id: Game identifier.
            step: Step number.
            summary: Summary dictionary of the step.
        """
        key = f"traj:{game_id}:{step}"
        self.set_memory(key, summary, tags=["trajectory", game_id])

    def recent_trajectory(self, game_id: str, k: int = 8) -> list[dict[str, Any]]:
        """Get recent trajectory steps for a game.

        Args:
            game_id: Game identifier.
            k: Number of recent steps to retrieve.

        Returns:
            List of trajectory summaries.
        """
        prefix = f"traj:{game_id}:"
        candidates = [
            (k2, e.value)
            for k2, e in self._store.items()
            if k2.startswith(prefix)
        ]
        candidates.sort(key=lambda x: x[0])
        return [v for _, v in candidates[-k:]]

    def store_hypothesis(
        self, game_id: str, hypothesis: str, confidence: float
    ) -> None:
        """Store a hypothesis for a game.

        Args:
            game_id: Game identifier.
            hypothesis: Hypothesis text.
            confidence: Confidence score.
        """
        key = (
            f"hyp:{game_id}:{hashlib.md5(hypothesis.encode()).hexdigest()[:8]}"
        )
        self.set_memory(
            key,
            {"text": hypothesis, "confidence": confidence},
            tags=["hypothesis", game_id],
        )

    def best_hypothesis(self, game_id: str) -> Optional[str]:
        """Get the best hypothesis for a game.

        Args:
            game_id: Game identifier.

        Returns:
            The hypothesis text with highest confidence, or None if no hypotheses exist.
        """
        prefix = f"hyp:{game_id}:"
        candidates = [
            e.value for k2, e in self._store.items() if k2.startswith(prefix)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x["confidence"])["text"]
