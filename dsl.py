"""DSL module for program generation, mutation, and execution."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

from arcengine import GameAction, GameState

ACTION_TOKENS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET", "WAIT"]
CONTROL_TOKENS = [
    "IF_CHANGED",
    "IF_BLOCKED",
    "REPEAT2",
    "REPEAT3",
    "EXPLORE",
    "END",
]
ALL_TOKENS = ACTION_TOKENS + CONTROL_TOKENS


@dataclass
class Program:
    """A program represented as a sequence of tokens."""

    tokens: list[str]

    def __str__(self) -> str:
        """Return string representation of the program."""
        return " ".join(self.tokens)

    def fingerprint(self) -> str:
        """Generate a unique fingerprint for the program."""
        return hashlib.sha256(json.dumps(self.tokens).encode()).hexdigest()[:16]


class ProgramGenerator:
    """Generator for creating and mutating programs."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the generator with a random seed.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = random.Random(seed)

    def sample(self, length: int = 6) -> Program:
        """Sample a random program.

        Args:
            length: Length of the program.

        Returns:
            A randomly generated program.
        """
        body = [
            self.rng.choice(ACTION_TOKENS * 2 + CONTROL_TOKENS)
            for _ in range(length - 1)
        ]
        body.append("END")
        return Program(body)

    def mutate(self, program: Program, n_mutations: int = 1) -> Program:
        """Mutate a program by randomly changing tokens.

        Args:
            program: Program to mutate.
            n_mutations: Number of mutations to apply.

        Returns:
            A mutated copy of the program.
        """
        tokens = program.tokens[:-1]  # Exclude END token
        for _ in range(n_mutations):
            if not tokens:
                break
            idx = self.rng.randint(0, len(tokens) - 1)
            tokens[idx] = self.rng.choice(ACTION_TOKENS * 2 + CONTROL_TOKENS[:-1])
        return Program(tokens + ["END"])

    def crossover(self, a: Program, b: Program) -> Program:
        """Create a new program by crossing over two programs.

        Args:
            a: First parent program.
            b: Second parent program.

        Returns:
            A child program combining parts of both parents.
        """
        split = self.rng.randint(1, min(len(a.tokens), len(b.tokens)) - 1)
        child_tokens = a.tokens[:split] + b.tokens[split:]
        if child_tokens[-1] != "END":
            child_tokens.append("END")
        return Program(child_tokens)


@dataclass
class ScoredProgram:
    """A program with an associated score."""

    program: Program
    score: float


def beam_search(
    scorer_fn: Callable[[Program], float],
    beam_width: int = 8,
    depth: int = 3,
    seed: int = 0,
    initial_programs: Optional[list[Program]] = None,
) -> list[ScoredProgram]:
    """Perform beam search to find high-scoring programs.

    Args:
        scorer_fn: Function to score a program.
        beam_width: Number of programs to keep in the beam.
        depth: Number of search iterations.
        seed: Random seed.
        initial_programs: Optional initial programs to seed the search.

    Returns:
        List of scored programs, sorted by score (highest first).
    """
    gen = ProgramGenerator(seed)
    if initial_programs:
        beam = [ScoredProgram(p, scorer_fn(p)) for p in initial_programs]
    else:
        candidates = [gen.sample() for _ in range(beam_width * 2)]
        beam = sorted(
            [ScoredProgram(p, scorer_fn(p)) for p in candidates],
            key=lambda x: -x.score,
        )[:beam_width]

    for _ in range(depth):
        children = []
        for sp in beam:
            children.append(gen.mutate(sp.program))
            children.append(gen.mutate(sp.program, n_mutations=2))
        if len(beam) >= 2:
            children.append(gen.crossover(beam[0].program, beam[1].program))
        scored = [ScoredProgram(p, scorer_fn(p)) for p in children]
        beam = sorted(beam + scored, key=lambda x: -x.score)[:beam_width]

    return beam


class Interpreter:
    """Interpreter for executing programs in an ARC environment."""

    def __init__(
        self,
        arc_env: Any,
        frames: list[Any],
        memory: Any,
        game_id: str,
    ) -> None:
        """Initialize the interpreter.

        Args:
            arc_env: ARC environment wrapper.
            frames: List of frames from the game.
            memory: TraceMemory instance for recording.
            game_id: Game identifier.
        """
        self._env = arc_env
        self._frames = frames
        self._memory = memory
        self._game_id = game_id

    def _step(self, token: str) -> tuple[float, bool]:
        """Execute a single action token.

        Args:
            token: Action token to execute.

        Returns:
            Tuple of (reward, done).
        """
        try:
            action = GameAction[token]
        except KeyError:
            return 0.0, False

        raw = self._env.step(action)

        # Convert raw frame to FrameData
        from arcengine import FrameData

        frame = FrameData(
            game_id=raw.game_id,
            frame=[
                arr.tolist() if hasattr(arr, "tolist") else arr
                for arr in raw.frame
            ],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )
        self._frames.append(frame)

        done = frame.state in (GameState.WIN, GameState.GAME_OVER)
        prev_levels = (
            self._frames[-2].levels_completed if len(self._frames) > 1 else 0
        )
        reward = float(frame.levels_completed - prev_levels)
        return reward, done

    def run(self, program: Program) -> dict[str, Any]:
        """Run a program and return execution summary.

        Args:
            program: Program to execute.

        Returns:
            Dictionary with execution results including reward, done status,
            delta_count, and action_trace.
        """
        total_reward, done, delta_count = 0.0, False, 0
        action_trace: list[str] = []
        prev_levels = (
            self._frames[-1].levels_completed if self._frames else 0
        )

        i, tokens = 0, program.tokens
        while i < len(tokens) and not done:
            tok = tokens[i]

            if tok == "END":
                break
            elif tok == "EXPLORE":
                # Random action from available actions
                available = (
                    self._frames[-1].available_actions if self._frames else []
                )
                if available:
                    rand_tok = str(random.choice(available)).split(".")[-1]
                else:
                    rand_tok = "ACTION1"
                r, done = self._step(rand_tok)
                total_reward += r
                action_trace.append(f"EXPLORE→{rand_tok}")
                delta_count += 1 if r != 0 else 0
            elif tok == "IF_CHANGED":
                if delta_count == 0:
                    i += 1  # Skip next instruction
                i += 1
                continue
            elif tok == "IF_BLOCKED":
                curr = self._frames[-1].levels_completed if self._frames else 0
                if curr == prev_levels:
                    i += 1  # Skip next instruction
                i += 1
                continue
            elif tok.startswith("REPEAT"):
                n = int(tok[-1])
                if i + 1 < len(tokens):
                    next_tok = tokens[i + 1]
                    for _ in range(n):
                        r, done = self._step(next_tok)
                        total_reward += r
                        action_trace.append(next_tok)
                        if done:
                            break
                    i += 2
                    continue
            else:
                # Execute action token
                r, done = self._step(tok)
                total_reward += r
                action_trace.append(tok)
                delta_count += 1 if r != 0 else 0

            i += 1

        # Record trajectory
        self._memory.record_trajectory(
            self._game_id,
            len(self._frames),
            {
                "program": str(program),
                "reward": total_reward,
                "done": done,
                "action_trace": action_trace,
            },
        )

        return {
            "reward": total_reward,
            "done": done,
            "delta_count": delta_count,
            "action_trace": action_trace,
        }
