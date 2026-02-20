"""Tests for animation frame interpolation selection logic."""

from agents.templates.claude_agent.claude_agents import select_animation_frames


class TestSelectAnimationFrames:
    """Test the interpolated frame selection for animations."""

    def _make_frames(self, n):
        """Create n dummy frames (each is a 1x1 grid with value = frame index)."""
        return [[[i]] for i in range(n)]

    def test_fewer_frames_than_max_returns_all(self):
        """When total frames <= max, return all frames with original indices."""
        frames = self._make_frames(3)
        result = select_animation_frames(frames, max_frames=7)
        assert len(result) == 3
        # Should be (original_index, frame) tuples
        assert result == [(0, [[0]]), (1, [[1]]), (2, [[2]])]

    def test_exact_max_returns_all(self):
        """When total frames == max, return all frames."""
        frames = self._make_frames(7)
        result = select_animation_frames(frames, max_frames=7)
        assert len(result) == 7
        indices = [idx for idx, _ in result]
        assert indices == [0, 1, 2, 3, 4, 5, 6]

    def test_first_and_last_always_included(self):
        """First and last frame must always be included regardless of count."""
        for total in [10, 20, 50, 70, 100]:
            frames = self._make_frames(total)
            result = select_animation_frames(frames, max_frames=7)
            indices = [idx for idx, _ in result]
            assert indices[0] == 0, f"First frame missing for total={total}"
            assert indices[-1] == total - 1, f"Last frame missing for total={total}"

    def test_max_frames_respected(self):
        """Never return more than max_frames."""
        for total in [8, 14, 21, 70, 100]:
            frames = self._make_frames(total)
            result = select_animation_frames(frames, max_frames=7)
            assert len(result) <= 7, (
                f"Too many frames for total={total}: got {len(result)}"
            )

    def test_indices_are_sorted(self):
        """Selected indices should be in ascending order."""
        frames = self._make_frames(70)
        result = select_animation_frames(frames, max_frames=7)
        indices = [idx for idx, _ in result]
        assert indices == sorted(indices)

    def test_no_duplicate_indices(self):
        """No duplicate frame indices should be selected."""
        for total in [8, 9, 10, 14, 70]:
            frames = self._make_frames(total)
            result = select_animation_frames(frames, max_frames=7)
            indices = [idx for idx, _ in result]
            assert len(indices) == len(set(indices)), (
                f"Duplicates for total={total}: {indices}"
            )

    def test_even_spacing_70_frames(self):
        """With 70 frames and max 7, indices should be roughly evenly spaced."""
        frames = self._make_frames(70)
        result = select_animation_frames(frames, max_frames=7)
        indices = [idx for idx, _ in result]
        assert len(indices) == 7
        assert indices[0] == 0
        assert indices[-1] == 69
        # Check roughly even spacing: gaps should be similar
        gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        # With 70 frames and 7 slots, ideal gap is ~11.5
        # All gaps should be between 10 and 13
        for gap in gaps:
            assert 8 <= gap <= 15, f"Gap {gap} too uneven in indices {indices}"

    def test_even_spacing_21_frames(self):
        """With 21 frames and max 7, check spacing."""
        frames = self._make_frames(21)
        result = select_animation_frames(frames, max_frames=7)
        indices = [idx for idx, _ in result]
        assert len(indices) == 7
        assert indices[0] == 0
        assert indices[-1] == 20

    def test_frame_data_matches_original(self):
        """The returned frame data should match the original at that index."""
        frames = self._make_frames(70)
        result = select_animation_frames(frames, max_frames=7)
        for idx, frame in result:
            assert frame == frames[idx], f"Frame data mismatch at index {idx}"

    def test_single_frame(self):
        """Single frame should just return that one frame."""
        frames = self._make_frames(1)
        result = select_animation_frames(frames, max_frames=7)
        assert len(result) == 1
        assert result == [(0, [[0]])]

    def test_two_frames(self):
        """Two frames should return both (first and last)."""
        frames = self._make_frames(2)
        result = select_animation_frames(frames, max_frames=7)
        assert len(result) == 2
        assert result == [(0, [[0]]), (1, [[1]])]

    def test_eight_frames_selects_seven(self):
        """Just over the limit: 8 frames with max 7 should select 7."""
        frames = self._make_frames(8)
        result = select_animation_frames(frames, max_frames=7)
        assert len(result) == 7
        indices = [idx for idx, _ in result]
        assert indices[0] == 0
        assert indices[-1] == 7

    def test_custom_max_frames(self):
        """Should work with different max_frames values."""
        frames = self._make_frames(20)
        result = select_animation_frames(frames, max_frames=5)
        assert len(result) == 5
        indices = [idx for idx, _ in result]
        assert indices[0] == 0
        assert indices[-1] == 19

    def test_empty_frames(self):
        """Empty frame list should return empty."""
        result = select_animation_frames([], max_frames=7)
        assert result == []
