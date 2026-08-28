import random

from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)

# Create sprites dictionary with all sprite definitions
sprites = {
    "sprite-1": Sprite(
        pixels=[
            [9],
        ],
        name="sprite-1",
        visible=True,
        collidable=True,
    ),
}

# Create levels array with all level definitions
levels = [
    # Level 1
    Level(
        sprites=[],
        grid_size=(8, 8),
    ),
    Level(
        sprites=[],
        grid_size=(16, 16),
    ),
    Level(
        sprites=[],
        grid_size=(24, 24),
    ),
    Level(
        sprites=[],
        grid_size=(32, 32),
    ),
    Level(
        sprites=[],
        grid_size=(64, 64),
    ),
]

BACKGROUND_COLOR = 0
PADDING_COLOR = 4

# Available colors for sprites (excluding all black -> white colors for visibility)
SPRITE_COLORS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Parameters for Procedural Generation
MIN_SIZE = 1
MAX_SIZE = 4


class Ab12(ARCBaseGame):
    """Click-to-remove puzzle game with seeded random sprite generation."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        # Create camera with background and padding colors
        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            width=8,
            height=8,
        )

        super().__init__(
            game_id="ab12",
            levels=levels,
            camera=camera,
        )

    def generate_sprites(self) -> None:
           """Generate a random set of sprites based on the seed."""
           # Determine number of sprites
           cell_count = self.current_level.grid_size[0] * self.current_level.grid_size[1]
           sprite_count = cell_count // 64  
           for idx in range(sprite_count):
               scale = self._rng.randint(MIN_SIZE, MAX_SIZE)
               color = self._rng.choice(SPRITE_COLORS)
               x = self._rng.randint(0, self.current_level.grid_size[0] - 1)
               y = self._rng.randint(0, self.current_level.grid_size[1] - 1)
               # Create the sprite setting color, scale, and position then add it to the level
               sprite = sprites[f"sprite-1"].clone().color_remap(None, color).set_scale(scale).set_position(x, y)
               self.current_level.add_sprite(sprite)

    def on_set_level(self, level: Level) -> None:
        """Called when the level is set."""
        # Generate sprites based on seed
        self.generate_sprites()

    def _check_win(self) -> bool:
        """Check if all targets have been removed."""
        return len(self.current_level._sprites) == 0

    def step(self) -> None:
        """Process game logic for each step."""
        # Handle click action (ACTION6)
        if self.action.id == GameAction.ACTION6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)

            # Convert display coordinates to grid coordinates
            coords = self.camera.display_to_grid(x, y)

            if coords:
                grid_x, grid_y = coords

                # Find and remove the clicked sprite from the level
                clicked_sprite = self.current_level.get_sprite_at(grid_x, grid_y)
                if clicked_sprite:
                    self.current_level.remove_sprite(clicked_sprite)

                    # Check win condition
                    if self._check_win():
                        self.next_level()

        self.complete_action()
