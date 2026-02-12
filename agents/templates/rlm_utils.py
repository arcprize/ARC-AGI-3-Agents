"""
RLM Utilities — Grid analysis helpers for the ARC-AGI-3 RLM Agent.

These functions are injected into the RLM's REPL environment so the LLM
can call them recursively to analyze game frames. All functions are pure
Python with no external dependencies beyond the standard library.
"""

from __future__ import annotations

from collections import deque
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Color semantics for ARC-AGI-3 environments
# ──────────────────────────────────────────────────────────────────────

COLOR_MAP: dict[int, str] = {
    0: "black/background",
    1: "dark_gray_1",
    2: "red",
    3: "dark_gray_2",
    4: "green",
    5: "pure_black",
    6: "blue",
    7: "yellow",
    8: "orange/floor",
    9: "purple",
    10: "white/wall",
    11: "gray/door_border",
    12: "dark_orange",
    13: "dark_red",
    14: "bright_green",
    15: "lavender",
}


def color_name(value: int) -> str:
    """Return a human-readable name for a grid integer value."""
    return COLOR_MAP.get(value, f"unknown({value})")


# ──────────────────────────────────────────────────────────────────────
# Grid comparison
# ──────────────────────────────────────────────────────────────────────


def diff_grids(
    prev: list[list[int]], curr: list[list[int]]
) -> list[dict[str, Any]]:
    """
    Compute pixel-level differences between two 2D grids.

    Returns a list of dicts:
        {"x": int, "y": int, "from": int, "to": int, "from_name": str, "to_name": str}
    """
    diffs: list[dict[str, Any]] = []
    if not prev or not curr:
        return diffs

    rows = min(len(prev), len(curr))
    for y in range(rows):
        cols = min(len(prev[y]), len(curr[y]))
        for x in range(cols):
            if prev[y][x] != curr[y][x]:
                diffs.append(
                    {
                        "x": x,
                        "y": y,
                        "from": prev[y][x],
                        "to": curr[y][x],
                        "from_name": color_name(prev[y][x]),
                        "to_name": color_name(curr[y][x]),
                    }
                )
    return diffs


def diff_summary(prev: list[list[int]], curr: list[list[int]]) -> str:
    """Produce a compact human-readable summary of grid changes."""
    diffs = diff_grids(prev, curr)
    if not diffs:
        return "No changes detected between frames."

    # Group by region
    min_x = min(d["x"] for d in diffs)
    max_x = max(d["x"] for d in diffs)
    min_y = min(d["y"] for d in diffs)
    max_y = max(d["y"] for d in diffs)

    # Count color transitions
    transitions: dict[str, int] = {}
    for d in diffs:
        key = f"{d['from_name']} -> {d['to_name']}"
        transitions[key] = transitions.get(key, 0) + 1

    lines = [
        f"Total changed pixels: {len(diffs)}",
        f"Changed region: x=[{min_x},{max_x}], y=[{min_y},{max_y}]",
        "Transitions:",
    ]
    for trans, count in sorted(transitions.items(), key=lambda t: -t[1]):
        lines.append(f"  {trans}: {count} pixels")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Object detection via flood-fill
# ──────────────────────────────────────────────────────────────────────


def find_objects(
    grid: list[list[int]],
    min_size: int = 2,
    ignore_colors: set[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Detect contiguous rectangular objects in a grid using flood-fill.

    Args:
        grid: 2D grid of ints
        min_size: minimum pixel count for an object
        ignore_colors: colors to skip (e.g., background/floor)

    Returns list of dicts:
        {"color": int, "color_name": str, "pixels": int,
         "bbox": {"x": int, "y": int, "w": int, "h": int},
         "center": {"x": float, "y": float}}
    """
    if ignore_colors is None:
        ignore_colors = {0, 8}  # background + floor by default

    if not grid or not grid[0]:
        return []

    rows = len(grid)
    cols = len(grid[0])
    visited: set[tuple[int, int]] = set()
    objects: list[dict[str, Any]] = []

    for y in range(rows):
        for x in range(cols):
            if (x, y) in visited or grid[y][x] in ignore_colors:
                continue

            color = grid[y][x]
            # BFS flood-fill for same color
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited.add((x, y))
            component: list[tuple[int, int]] = []

            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < cols
                        and 0 <= ny < rows
                        and (nx, ny) not in visited
                        and grid[ny][nx] == color
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny))

            if len(component) >= min_size:
                xs = [p[0] for p in component]
                ys = [p[1] for p in component]
                objects.append(
                    {
                        "color": color,
                        "color_name": color_name(color),
                        "pixels": len(component),
                        "bbox": {
                            "x": min(xs),
                            "y": min(ys),
                            "w": max(xs) - min(xs) + 1,
                            "h": max(ys) - min(ys) + 1,
                        },
                        "center": {
                            "x": sum(xs) / len(xs),
                            "y": sum(ys) / len(ys),
                        },
                    }
                )

    return sorted(objects, key=lambda o: -o["pixels"])


# ──────────────────────────────────────────────────────────────────────
# Player / Key / Door / Energy detection
# ──────────────────────────────────────────────────────────────────────


def find_player(grid: list[list[int]]) -> dict[str, Any] | None:
    """
    Locate the player sprite in the grid.

    The player is typically a 3x3 or 4x4 pattern containing green (4)
    and black (0) pixels in the upper portion.

    Returns: {"x": int, "y": int} center position, or None if not found.
    """
    if not grid or not grid[0]:
        return None

    rows = len(grid)
    cols = len(grid[0])

    # Look for a cluster of green (4) pixels that forms the player body
    green_objects = find_objects(grid, min_size=4, ignore_colors={0, 8, 10, 11})
    for obj in green_objects:
        if obj["color"] == 4 and 4 <= obj["pixels"] <= 16:
            bbox = obj["bbox"]
            # Player is small-ish, 3x3 or 4x4
            if bbox["w"] <= 5 and bbox["h"] <= 5:
                return {
                    "x": int(obj["center"]["x"]),
                    "y": int(obj["center"]["y"]),
                    "bbox": bbox,
                }

    return None


def find_door(grid: list[list[int]]) -> dict[str, Any] | None:
    """
    Find the exit door — a 4x4 square with INT<11> (gray) border.

    Returns: {"x": int, "y": int, "bbox": {...}, "inner_pattern": [[int]]}
    """
    if not grid or not grid[0]:
        return None

    rows = len(grid)
    cols = len(grid[0])

    # Search for 4x4 regions with gray (11) border
    for y in range(rows - 3):
        for x in range(cols - 3):
            # Check if top and bottom rows are all 11
            top_border = all(grid[y][x + dx] == 11 for dx in range(4))
            bottom_border = all(grid[y + 3][x + dx] == 11 for dx in range(4))
            left_border = all(grid[y + dy][x] == 11 for dy in range(4))
            right_border = all(grid[y + dy][x + 3] == 11 for dy in range(4))

            if top_border and bottom_border and left_border and right_border:
                # Extract inner 2x2 pattern
                inner = [
                    [grid[y + 1][x + 1], grid[y + 1][x + 2]],
                    [grid[y + 2][x + 1], grid[y + 2][x + 2]],
                ]
                return {
                    "x": x,
                    "y": y,
                    "bbox": {"x": x, "y": y, "w": 4, "h": 4},
                    "inner_pattern": inner,
                }

    return None


def find_key(grid: list[list[int]], grid_size: int = 64) -> list[list[int]] | None:
    """
    Extract the key pattern from the bottom-left corner of the grid.

    The key is usually a 6x6 square in the bottom-left.

    Returns: The 6x6 key pattern as a 2D list, or None.
    """
    if not grid or len(grid) < grid_size:
        return None

    key_size = 6
    start_y = grid_size - key_size
    start_x = 0

    try:
        key_pattern = []
        for y in range(start_y, start_y + key_size):
            row = []
            for x in range(start_x, start_x + key_size):
                row.append(grid[y][x])
            key_pattern.append(row)
        return key_pattern
    except IndexError:
        return None


def extract_energy(grid: list[list[int]], grid_size: int = 64) -> dict[str, int]:
    """
    Read energy indicators from the grid.

    Energy is shown on the 3rd row as blue (6) for unused and orange (8) for used.

    Returns: {"unused": int, "used": int, "total": int}
    """
    if not grid or len(grid) < 3 or len(grid[0]) < grid_size:
        return {"unused": 0, "used": 0, "total": 0}

    # Energy is typically on row 2 (0-indexed), in specific columns
    energy_row = grid[2] if len(grid) > 2 else []
    unused = sum(1 for v in energy_row if v == 6)
    used = sum(1 for v in energy_row if v == 8)

    return {"unused": unused, "used": used, "total": unused + used}


def extract_lives(grid: list[list[int]], grid_size: int = 64) -> int:
    """
    Count remaining lives from the top-right corner of the grid.

    Lives are 2x2 red (2) squares in the top-right area.

    Returns: number of lives remaining.
    """
    if not grid or len(grid) < 4:
        return 0

    lives = 0
    # Check rows 1-2, last ~10 columns for 2x2 red blocks
    for y in range(min(3, len(grid) - 1)):
        for x in range(max(0, len(grid[0]) - 10), len(grid[0]) - 1):
            if (
                grid[y][x] == 2
                and grid[y][x + 1] == 2
                and y + 1 < len(grid)
                and grid[y + 1][x] == 2
                and grid[y + 1][x + 1] == 2
            ):
                lives += 1

    return lives


# ──────────────────────────────────────────────────────────────────────
# Grid summary
# ──────────────────────────────────────────────────────────────────────


def summarize_grid(grid: list[list[int]], grid_size: int = 64) -> str:
    """
    Produce a compact text summary of the current game grid.

    Includes player position, door location, key pattern, energy, and
    a high-level description of the grid layout.
    """
    lines: list[str] = []

    player = find_player(grid)
    door = find_door(grid)
    key = find_key(grid, grid_size)
    energy = extract_energy(grid, grid_size)
    lives = extract_lives(grid, grid_size)

    if player:
        lines.append(f"Player at: ({player['x']}, {player['y']})")
    else:
        lines.append("Player: not found")

    if door:
        lines.append(
            f"Door at: ({door['x']}, {door['y']}), "
            f"inner pattern: {door['inner_pattern']}"
        )
    else:
        lines.append("Door: not found")

    if key:
        lines.append(f"Key pattern (6x6 from bottom-left): {key}")
    else:
        lines.append("Key: not found")

    lines.append(
        f"Energy: {energy['unused']} unused / {energy['used']} used "
        f"(total: {energy['total']})"
    )
    lines.append(f"Lives remaining: {lives}")

    # Count unique colors
    color_counts: dict[int, int] = {}
    for row in grid:
        for val in row:
            color_counts[val] = color_counts.get(val, 0) + 1

    lines.append("Color distribution:")
    for c, count in sorted(color_counts.items(), key=lambda t: -t[1])[:8]:
        pct = count / (grid_size * grid_size) * 100
        lines.append(f"  {color_name(c)} ({c}): {count} px ({pct:.1f}%)")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# REPL injection helper
# ──────────────────────────────────────────────────────────────────────


def get_repl_namespace() -> dict[str, Any]:
    """
    Return a namespace dict containing all utility functions,
    ready to be injected into the RLM's REPL environment.
    """
    return {
        "color_name": color_name,
        "color_map": COLOR_MAP,
        "diff_grids": diff_grids,
        "diff_summary": diff_summary,
        "find_objects": find_objects,
        "find_player": find_player,
        "find_door": find_door,
        "find_key": find_key,
        "extract_energy": extract_energy,
        "extract_lives": extract_lives,
        "summarize_grid": summarize_grid,
    }
