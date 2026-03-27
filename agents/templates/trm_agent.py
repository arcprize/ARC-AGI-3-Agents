"""TRM Agent implementation using program synthesis and trace-based reasoning."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent
from dsl import (
    ALL_TOKENS,
    ACTION_TOKENS,
    Program,
    ProgramGenerator,
    beam_search,
)
from memory import TraceMemory

logger = logging.getLogger(__name__)


class ErrorEncoder(nn.Module):
    """Neural encoder for error/program tokens."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """Initialize the error encoder.

        Args:
            d_model: Model dimension.
            vocab_size: Size of the token vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, token_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode token IDs.

        Args:
            token_ids: Optional tensor of token IDs.

        Returns:
            Encoded representation.
        """
        device = self.embedding.weight.device
        if token_ids is None or token_ids.numel() == 0:
            return torch.zeros(1, 1, self.d_model, device=device)
        emb = self.embedding(token_ids.to(device))
        _, h = self.gru(emb)
        return h.permute(1, 0, 2)


class ObservationEncoder(nn.Module):
    """Neural encoder for game observations."""

    MAX_CELLS = 256

    def __init__(self, d_model: int) -> None:
        """Initialize the observation encoder.

        Args:
            d_model: Model dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(self.MAX_CELLS, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def _frame_to_vec(self, frame_layers: list[Any]) -> torch.Tensor:
        """Convert frame layers to a fixed-size vector.

        Args:
            frame_layers: List of frame layers.

        Returns:
            Fixed-size tensor representation.
        """
        if not frame_layers:
            return torch.zeros(self.MAX_CELLS)

        flat = np.concatenate(
            [np.array(layer).flatten() for layer in frame_layers]
        ).astype(np.float32)
        flat = flat / 10.0

        if len(flat) >= self.MAX_CELLS:
            idx = np.linspace(0, len(flat) - 1, self.MAX_CELLS).astype(int)
            flat = flat[idx]
        else:
            flat = np.pad(flat, (0, self.MAX_CELLS - len(flat)))

        return torch.tensor(flat, dtype=torch.float32)

    def forward(self, frame: FrameData) -> torch.Tensor:
        """Encode a frame.

        Args:
            frame: Frame data to encode.

        Returns:
            Encoded representation.
        """
        vec = self._frame_to_vec(frame.frame if frame.frame else [])
        out = self.proj(vec.unsqueeze(0))
        return self.layer_norm(out).unsqueeze(1)


class TRM(nn.Module):
    """Trace-based Reasoning Model."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        vocab_size: int = len(ALL_TOKENS) + 2,
        max_recursion: int = 6,
    ) -> None:
        """Initialize the TRM model.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            vocab_size: Size of the token vocabulary.
            max_recursion: Maximum recursion depth.
        """
        super().__init__()
        self.d_model = d_model
        self.max_recursion = max_recursion
        self.vocab_size = vocab_size

        self.obs_enc = ObservationEncoder(d_model)
        self.err_enc = ErrorEncoder(d_model, vocab_size)
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.0,
        )
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)

        self.tok2idx: dict[str, int] = {
            t: i + 1 for i, t in enumerate(ALL_TOKENS)
        }
        self.idx2tok: dict[int, str] = {v: k for k, v in self.tok2idx.items()}

    def encode_program(self, program: Program) -> torch.Tensor:
        """Encode a program to token IDs.

        Args:
            program: Program to encode.

        Returns:
            Tensor of token IDs.
        """
        ids = [self.tok2idx.get(t, 0) for t in program.tokens]
        return torch.tensor([ids], dtype=torch.long)

    def decode_policy(
        self, z: torch.Tensor, temperature: float = 1.0
    ) -> Program:
        """Decode a policy from a latent representation.

        Args:
            z: Latent representation.
            temperature: Sampling temperature.

        Returns:
            Decoded program.
        """
        tokens: list[str] = []
        hidden = z

        for _ in range(8):
            logits = self.policy_head(hidden[:, -1]) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            tok = self.idx2tok.get(int(idx), "END")
            tokens.append(tok)

            if tok == "END":
                break

            # Update hidden state with new token
            next_emb = self.err_enc.embedding(
                torch.tensor([[int(idx)]])
            ).to(z.device)
            hidden = torch.cat([hidden, next_emb], dim=1)

        return Program(tokens if tokens else ["EXPLORE", "END"])

    def forward(
        self,
        latest_frame: FrameData,
        error_program: Optional[Program] = None,
        temperature: float = 1.0,
    ) -> tuple[Program, float]:
        """Forward pass to generate a program and value estimate.

        Args:
            latest_frame: Latest frame from the game.
            error_program: Optional program that led to an error.
            temperature: Sampling temperature.

        Returns:
            Tuple of (generated program, value estimate).
        """
        obs = self.obs_enc(latest_frame)
        err = self.err_enc(
            self.encode_program(error_program) if error_program else None
        )
        z = obs + err

        for _ in range(self.max_recursion):
            z = self.block(z)

        program = self.decode_policy(z, temperature)
        value = self.value_head(z[:, 0]).squeeze(-1).item()

        return program, float(value)


class TRMAgent(Agent):
    """Agent using Trace-based Reasoning Model for program synthesis."""

    CURIOSITY_STEPS: int = 12
    BEAM_WIDTH: int = 6
    BEAM_DEPTH: int = 2
    TRM_TEMPERATURE: float = 1.2

    def __init__(
        self, *args: Any, memory_path: str = "pmll_memory.jsonl", **kwargs: Any
    ) -> None:
        """Initialize the TRM agent.

        Args:
            memory_path: Path to the memory file.
            *args: Positional arguments for Agent base class.
            **kwargs: Keyword arguments for Agent base class.
        """
        super().__init__(*args, **kwargs)
        self.memory = TraceMemory(path=memory_path)
        self.trm = TRM()
        self.trm.eval()
        self.gen = ProgramGenerator(seed=42)

        self._last_error_program: Optional[Program] = None
        self._prev_levels: int = 0
        self._last_program: Optional[Program] = None
        self._queued_actions: list[GameAction] = []

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Check if the agent is done.

        Args:
            frames: List of all frames.
            latest_frame: The most recent frame.

        Returns:
            True if the game is won or over.
        """
        return latest_frame.state in (GameState.WIN, GameState.GAME_OVER)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose the next action.

        Args:
            frames: List of all frames.
            latest_frame: The most recent frame.

        Returns:
            The chosen game action.
        """
        # If we have queued actions, use them first
        if self._queued_actions:
            return self._queued_actions.pop(0)

        # Choose strategy based on action count
        if self.action_counter < self.CURIOSITY_STEPS:
            program = self._curiosity_program(latest_frame)
        else:
            program = self._synthesis_program(frames, latest_frame)

        self._last_program = program

        # Extract action tokens and queue them
        action_tokens = [t for t in program.tokens if t in ACTION_TOKENS]
        for tok in action_tokens[1:]:
            try:
                self._queued_actions.append(GameAction[tok])
            except KeyError:
                pass

        # Return the first action
        first_tok = action_tokens[0] if action_tokens else "ACTION1"
        try:
            return GameAction[first_tok]
        except KeyError:
            return GameAction.ACTION1

    def _curiosity_program(self, latest_frame: FrameData) -> Program:
        """Generate a curiosity-driven program.

        Args:
            latest_frame: The most recent frame.

        Returns:
            A simple random program for exploration.
        """
        return self.gen.sample(length=3)

    def _synthesis_program(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> Program:
        """Synthesize a program using TRM and beam search.

        Args:
            frames: List of all frames.
            latest_frame: The most recent frame.

        Returns:
            A synthesized program.
        """
        # Generate initial program using TRM
        with torch.no_grad():
            trm_program, _ = self.trm(
                latest_frame,
                error_program=self._last_error_program,
                temperature=self.TRM_TEMPERATURE,
            )

        # Score programs using heuristics
        def heuristic_score(program: Program) -> float:
            if self.memory.is_program_hopeless(program.tokens):
                return -999.0
            # Prefer programs with diverse actions
            return len(set(t for t in program.tokens if t in ACTION_TOKENS))

        # Perform beam search
        beam = beam_search(
            scorer_fn=heuristic_score,
            beam_width=self.BEAM_WIDTH,
            depth=self.BEAM_DEPTH,
            seed=self.action_counter,
            initial_programs=[trm_program, self.gen.sample()],
        )

        best = beam[0].program

        # Periodically store hypothesis
        if self.action_counter % 5 == 0:
            self.memory.store_hypothesis(
                self.game_id,
                f"Best program: {best}",
                confidence=beam[0].score / 10.0,
            )

        return best

    def append_frame(self, frame: FrameData) -> None:
        """Append a frame and update memory.

        Args:
            frame: Frame to append.
        """
        super().append_frame(frame)

        # Calculate reward
        reward_delta = frame.levels_completed - self._prev_levels
        self._prev_levels = frame.levels_completed

        # Record trajectory
        self.memory.record_trajectory(
            self.game_id,
            self.action_counter,
            {"program": str(self._last_program), "reward": reward_delta},
        )

        # Mark failed programs
        if reward_delta == 0 and self.action_counter > self.CURIOSITY_STEPS:
            if self._last_program:
                self.memory.mark_failed_program(
                    self._last_program.tokens, "no_progress"
                )
                self._last_error_program = self._last_program
