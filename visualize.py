#!/usr/bin/env python3
"""
ARC-AGI Recording Visualizer

A comprehensive visualization tool for ARC-AGI agent recordings using matplotlib.
Supports both static frame visualization and animated gameplay sequences.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ARC color palette - standard 10 colors used in ARC challenges
ARC_COLORS = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Grey
    "#F012BE",  # 6: Magenta
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Sky Blue
    "#870C25",  # 9: Brown
    "#FFFFFF",  # 10: White (for background if needed)
    "#001F3F",  # 11: Navy
    "#39CCCC",  # 12: Teal
    "#01FF70",  # 13: Lime
    "#85144B",  # 14: Maroon
    "#B10DC9",  # 15: Purple
]


class ARCVisualizer:
    """Main visualization class for ARC-AGI recordings."""

    def __init__(self, recording_file: str) -> None:
        """Initialize visualizer with recording file."""
        self.recording_file = Path(recording_file)
        self.frames: List[np.ndarray] = []
        self.timestamps: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.scores: List[int] = []
        self.game_states: List[str] = []

        # Set up color map
        self.cmap = ListedColormap(ARC_COLORS[:16])  # Support up to 16 colors

        self._load_recording()

    def _load_recording(self) -> None:
        """Load and parse the recording file."""
        print(f"Loading recording: {self.recording_file}")

        with open(self.recording_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())

                    # Extract timestamp
                    timestamp = entry.get("timestamp", "")
                    self.timestamps.append(timestamp)

                    # Extract game data
                    data = entry.get("data", {})

                    # Extract frame if present
                    if "frame" in data and data["frame"]:
                        frame = np.array(data["frame"][0])  # First element is the grid
                        self.frames.append(frame)

                    # Extract action info
                    action_input = data.get("action_input", {})
                    self.actions.append(action_input)

                    # Extract score and state
                    self.scores.append(data.get("score", 0))
                    self.game_states.append(data.get("state", "UNKNOWN"))

                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")

        print(f"Loaded {len(self.frames)} frames from recording")

    def plot_frame(
        self, frame_idx: int, ax: Optional[plt.Axes] = None, title: Optional[str] = None
    ) -> plt.Axes:
        """Plot a single frame."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if frame_idx >= len(self.frames):
            ax.text(
                0.5,
                0.5,
                f"Frame {frame_idx} not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        frame = self.frames[frame_idx]

        # Create the visualization
        ax.imshow(frame, cmap=self.cmap, vmin=0, vmax=15, aspect="equal")

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, frame.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, frame.shape[0], 1), minor=True)
        ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.5, alpha=0.3)

        # Remove major ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set title
        if title is None:
            action = self.actions[frame_idx] if frame_idx < len(self.actions) else {}
            action_type = action.get("id", "Unknown")
            score = self.scores[frame_idx] if frame_idx < len(self.scores) else 0
            timestamp = (
                self.timestamps[frame_idx]
                if frame_idx < len(self.timestamps)
                else "Unknown"
            )
            title = (
                f"Frame {frame_idx}: Action {action_type} | Score: {score}\n{timestamp}"
            )

        ax.set_title(title, fontsize=12, pad=20)

        return ax

    def plot_sequence(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        max_frames: int = 12,
    ) -> plt.Figure:
        """Plot a sequence of frames in a grid."""
        if end_frame is None:
            end_frame = min(len(self.frames), start_frame + max_frames)

        frames_to_show = min(end_frame - start_frame, max_frames)

        # Calculate grid dimensions
        cols = min(4, frames_to_show)
        rows = (frames_to_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(frames_to_show):
            frame_idx = start_frame + i
            self.plot_frame(frame_idx, axes[i])

        # Hide unused subplots
        for i in range(frames_to_show, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f"ARC-AGI Game Sequence: Frames {start_frame}-{end_frame - 1}",
            fontsize=16,
            y=0.98,
        )
        fig.tight_layout()

        return fig

    def create_animation(
        self, interval: int = 1000, save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """Create an animated visualization of the entire game sequence."""
        if not self.frames:
            raise ValueError("No frames to animate")

        fig, ax = plt.subplots(figsize=(12, 10))

        def animate(frame_idx: int) -> None:
            ax.clear()
            self.plot_frame(frame_idx, ax)

            # Add progress info
            progress_text = f"Frame {frame_idx + 1}/{len(self.frames)}"
            if frame_idx < len(self.actions):
                action = self.actions[frame_idx]
                if isinstance(action, dict):
                    action_info = f"Action: {action.get('id', 'Unknown')}"
                    if "x" in action.get("data", {}) and "y" in action.get("data", {}):
                        x, y = action["data"]["x"], action["data"]["y"]
                        action_info += f" at ({x}, {y})"
                    progress_text += f" | {action_info}"

            ax.text(
                0.02,
                0.98,
                progress_text,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment="top",
                fontsize=10,
            )

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(self.frames),
            interval=interval,
            repeat=True,
            blit=False,
        )

        if save_path:
            print(f"Saving animation to {save_path}")
            anim.save(save_path, writer="pillow", fps=1000 // interval)

        return anim

    def plot_score_progression(self) -> plt.Figure:
        """Plot the score progression over time."""
        fig, ax = plt.subplots(figsize=(12, 6))

        frames = range(len(self.scores))
        ax.plot(frames, self.scores, marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Score")
        ax.set_title("Score Progression Throughout Game")
        ax.grid(True, alpha=0.3)

        # Highlight score changes
        for i in range(1, len(self.scores)):
            if self.scores[i] != self.scores[i - 1]:
                ax.axvline(x=i, color="red", linestyle="--", alpha=0.5)
                ax.annotate(
                    f"+{self.scores[i] - self.scores[i - 1]}",
                    xy=(i, self.scores[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

        return fig

    def analyze_actions(self) -> Dict[str, Any]:
        """Analyze the actions taken during the game."""
        action_counts: Dict[str, int] = {}
        coordinate_actions = []

        for action in self.actions:
            if isinstance(action, dict):
                action_id = action.get("id", "Unknown")
                action_counts[action_id] = action_counts.get(action_id, 0) + 1

                # Track coordinate-based actions
                if "data" in action and isinstance(action["data"], dict):
                    data = action["data"]
                    if "x" in data and "y" in data:
                        coordinate_actions.append((data["x"], data["y"], action_id))

        return {
            "action_counts": action_counts,
            "coordinate_actions": coordinate_actions,
            "total_actions": len(self.actions),
        }

    def plot_action_heatmap(self) -> plt.Figure:
        """Create a heatmap of where actions were taken."""
        analysis = self.analyze_actions()
        coordinate_actions = analysis["coordinate_actions"]

        if not coordinate_actions or not self.frames:
            print("No coordinate-based actions found")
            return plt.figure()

        # Get grid dimensions from first frame
        grid_shape = self.frames[0].shape
        action_grid = np.zeros(grid_shape)

        # Count actions at each position
        for x, y, action_id in coordinate_actions:
            if 0 <= y < grid_shape[0] and 0 <= x < grid_shape[1]:
                action_grid[y, x] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Show first frame as reference
        self.plot_frame(0, ax1, "Initial Game State")

        # Show action heatmap
        im = ax2.imshow(action_grid, cmap="Reds", alpha=0.8)
        ax2.set_title("Action Heatmap")
        ax2.set_xlabel("X Coordinate")
        ax2.set_ylabel("Y Coordinate")

        # Add colorbar
        plt.colorbar(im, ax=ax2, label="Number of Actions")

        # Add grid
        ax2.set_xticks(np.arange(-0.5, grid_shape[1], 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, grid_shape[0], 1), minor=True)
        ax2.grid(which="minor", color="grey", linestyle="-", linewidth=0.5, alpha=0.3)

        return fig

    def generate_summary_report(self, output_dir: str = "visualization_output") -> None:
        """Generate a comprehensive summary report with all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Generating summary report in {output_path}")

        # Analysis
        analysis = self.analyze_actions()

        # 1. Save frame sequence
        if self.frames:
            seq_fig = self.plot_sequence()
            seq_fig.savefig(
                output_path / "frame_sequence.png", dpi=150, bbox_inches="tight"
            )
            plt.close(seq_fig)
            print("✓ Saved frame sequence")

        # 2. Save score progression
        if self.scores:
            score_fig = self.plot_score_progression()
            score_fig.savefig(
                output_path / "score_progression.png", dpi=150, bbox_inches="tight"
            )
            plt.close(score_fig)
            print("✓ Saved score progression")

        # 3. Save action heatmap
        heatmap_fig = self.plot_action_heatmap()
        heatmap_fig.savefig(
            output_path / "action_heatmap.png", dpi=150, bbox_inches="tight"
        )
        plt.close(heatmap_fig)
        print("✓ Saved action heatmap")

        # 4. Create and save animation
        try:
            anim = self.create_animation(interval=1500)
            anim.save(output_path / "game_animation.gif", writer="pillow", fps=0.67)
            print("✓ Saved game animation")
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")

        # 5. Save text summary
        with open(output_path / "analysis_summary.txt", "w") as f:
            f.write("ARC-AGI Recording Analysis Summary\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Recording file: {self.recording_file}\n")
            f.write(f"Total frames: {len(self.frames)}\n")
            f.write(f"Total actions: {analysis['total_actions']}\n")
            f.write(f"Final score: {self.scores[-1] if self.scores else 'N/A'}\n")
            f.write(
                f"Final state: {self.game_states[-1] if self.game_states else 'N/A'}\n\n"
            )

            f.write("Action breakdown:\n")
            for action_id, count in analysis["action_counts"].items():
                f.write(f"  Action {action_id}: {count} times\n")

            if analysis["coordinate_actions"]:
                f.write(
                    f"\nCoordinate-based actions: {len(analysis['coordinate_actions'])}\n"
                )

        print("✓ Saved analysis summary")
        print(f"\nSummary report generated in: {output_path}")


def main() -> int:
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize ARC-AGI recording files")
    parser.add_argument("recording_file", help="Path to the recording JSONL file")
    parser.add_argument(
        "--output",
        "-o",
        default="visualization_output",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Show specific frame range (start, end)",
    )
    parser.add_argument(
        "--animate", "-a", action="store_true", help="Create and show animation"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=1000,
        help="Animation interval in milliseconds",
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="Skip generating full summary report"
    )

    args = parser.parse_args()

    # Initialize visualizer
    try:
        visualizer = ARCVisualizer(args.recording_file)
    except FileNotFoundError:
        print(f"Error: Recording file '{args.recording_file}' not found")
        return 1
    except Exception as e:
        print(f"Error loading recording: {e}")
        return 1

    if not visualizer.frames:
        print("Warning: No frames found in recording")
        return 1

    # Generate summary report unless disabled
    if not args.no_summary:
        visualizer.generate_summary_report(args.output)

    # Show specific frames if requested
    if args.frames:
        start, end = args.frames
        visualizer.plot_sequence(start, end)
        plt.show()

    # Create animation if requested
    if args.animate:
        visualizer.create_animation(args.interval)
        plt.show()

    return 0


if __name__ == "__main__":
    exit(main())
