"""Tests for memory, dsl, and trm_agent modules."""

import os
import tempfile

import pytest

from dsl import (
    ACTION_TOKENS,
    CONTROL_TOKENS,
    Program,
    ProgramGenerator,
    ScoredProgram,
    beam_search,
)
from memory import MemoryEntry, TraceMemory


@pytest.mark.unit
class TestMemory:
    """Test the TraceMemory class."""

    def test_memory_init(self):
        """Test memory initialization."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name

        try:
            memory = TraceMemory(path=path)
            assert memory._path.name == os.path.basename(path)
            assert len(memory._store) == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_memory_set_get(self):
        """Test setting and getting memory entries."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name

        try:
            memory = TraceMemory(path=path)

            # Set and get a value
            memory.set_memory("test_key", {"value": 42}, tags=["test"])
            result = memory.get_memory("test_key")
            assert result == {"value": 42}

            # Get non-existent key
            assert memory.get_memory("nonexistent") is None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_memory_persistence(self):
        """Test that memory persists across instances."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name

        try:
            # Create memory and set a value
            memory1 = TraceMemory(path=path)
            memory1.set_memory("persist_key", "persistent_value", tags=["persist"])

            # Create new instance and check value persists
            memory2 = TraceMemory(path=path)
            result = memory2.get_memory("persist_key")
            assert result == "persistent_value"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_memory_delete(self):
        """Test deleting memory entries."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name

        try:
            memory = TraceMemory(path=path)
            memory.set_memory("delete_key", "value")

            # Delete existing key
            assert memory.delete_memory("delete_key") is True
            assert memory.get_memory("delete_key") is None

            # Delete non-existent key
            assert memory.delete_memory("nonexistent") is False
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_list_memories(self):
        """Test listing memory entries."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name

        try:
            memory = TraceMemory(path=path)
            memory.set_memory("key1", "value1", tags=["tag1"])
            memory.set_memory("key2", "value2", tags=["tag2"])
            memory.set_memory("key3", "value3", tags=["tag1", "tag2"])

            # List all
            all_memories = memory.list_memories()
            assert len(all_memories) == 3

            # List by tag
            tag1_memories = memory.list_memories(tag="tag1")
            assert len(tag1_memories) == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_program_fingerprint(self):
        """Test program fingerprinting."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name

        try:
            memory = TraceMemory(path=path)

            # Same tokens should produce same fingerprint
            fp1 = memory.program_fingerprint(["ACTION1", "ACTION2", "END"])
            fp2 = memory.program_fingerprint(["ACTION1", "ACTION2", "END"])
            assert fp1 == fp2

            # Different tokens should produce different fingerprint
            fp3 = memory.program_fingerprint(["ACTION3", "ACTION4", "END"])
            assert fp1 != fp3

            # Order doesn't matter (sorted internally)
            fp4 = memory.program_fingerprint(["ACTION2", "ACTION1", "END"])
            assert fp1 == fp4
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_failed_program_tracking(self):
        """Test tracking failed programs."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = f.name

        try:
            memory = TraceMemory(path=path)
            tokens = ["ACTION1", "RESET", "END"]

            # Mark as failed multiple times
            assert not memory.is_program_hopeless(tokens)
            memory.mark_failed_program(tokens, "error1")
            assert not memory.is_program_hopeless(tokens)
            memory.mark_failed_program(tokens, "error2")
            assert not memory.is_program_hopeless(tokens)
            memory.mark_failed_program(tokens, "error3")
            assert memory.is_program_hopeless(tokens)
        finally:
            if os.path.exists(path):
                os.unlink(path)


@pytest.mark.unit
class TestDSL:
    """Test the DSL module."""

    def test_program_creation(self):
        """Test Program creation."""
        tokens = ["ACTION1", "ACTION2", "END"]
        program = Program(tokens=tokens)
        assert program.tokens == tokens
        assert str(program) == "ACTION1 ACTION2 END"

    def test_program_fingerprint(self):
        """Test Program fingerprint."""
        p1 = Program(tokens=["ACTION1", "END"])
        p2 = Program(tokens=["ACTION1", "END"])
        p3 = Program(tokens=["ACTION2", "END"])

        assert p1.fingerprint() == p2.fingerprint()
        assert p1.fingerprint() != p3.fingerprint()

    def test_program_generator_sample(self):
        """Test ProgramGenerator sampling."""
        gen = ProgramGenerator(seed=42)
        program = gen.sample(length=5)

        assert len(program.tokens) == 5
        assert program.tokens[-1] == "END"
        assert all(t in ACTION_TOKENS + CONTROL_TOKENS for t in program.tokens)

    def test_program_generator_mutate(self):
        """Test ProgramGenerator mutation."""
        gen = ProgramGenerator(seed=42)
        original = Program(tokens=["ACTION1", "ACTION2", "ACTION3", "END"])
        mutated = gen.mutate(original, n_mutations=1)

        # Should still end with END
        assert mutated.tokens[-1] == "END"
        # Should have same length
        assert len(mutated.tokens) == len(original.tokens)
        # Should be different from original
        assert mutated.tokens != original.tokens

    def test_program_generator_crossover(self):
        """Test ProgramGenerator crossover."""
        gen = ProgramGenerator(seed=42)
        p1 = Program(tokens=["ACTION1", "ACTION2", "ACTION3", "END"])
        p2 = Program(tokens=["ACTION4", "RESET", "WAIT", "END"])
        child = gen.crossover(p1, p2)

        # Should end with END
        assert child.tokens[-1] == "END"
        # Should have valid tokens
        assert all(t in ACTION_TOKENS + CONTROL_TOKENS for t in child.tokens)

    def test_beam_search(self):
        """Test beam search."""

        def simple_scorer(program: Program) -> float:
            # Score based on number of unique action tokens
            actions = [t for t in program.tokens if t in ACTION_TOKENS]
            return len(set(actions))

        results = beam_search(
            scorer_fn=simple_scorer, beam_width=3, depth=2, seed=42
        )

        # Should return list of ScoredPrograms
        assert len(results) > 0
        assert all(isinstance(sp, ScoredProgram) for sp in results)
        # Should be sorted by score (descending)
        scores = [sp.score for sp in results]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
class TestTRMAgent:
    """Test the TRM Agent."""

    def test_trm_agent_import(self):
        """Test that TRM agent can be imported."""
        from agents.templates.trm_agent import TRMAgent

        assert TRMAgent is not None

    def test_trm_agent_registration(self):
        """Test that TRM agent is registered."""
        from agents import AVAILABLE_AGENTS

        assert "trmagent" in AVAILABLE_AGENTS
