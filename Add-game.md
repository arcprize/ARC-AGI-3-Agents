> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

> # Create ARC-AGI-3 Environment

> Building and adding a game to ARC-AGI-3 Environments

This example walks through a simple game where randomly generated sprites appear and the player must click to remove them all to win.

> [See the full code](#complete-code)

> ## Game File Structure

Each game adheres to the following structure:

* **Imports**
  <br />Import ARCEngine classes and any standard library modules needed.
* **Sprite Definitions**
  <br />Define sprite templates in a dictionary. Sprites are objects with a name, rectangular pixel array, and optional tags.
* **Level Definitions**
  <br />Define levels as a list of `Level` objects, each containing sprite placements and configuration data.
* **Constants**
  <br />Define game-wide constants for colors, grid size, and gameplay parameters.
* **Game Class**
  <br />Create a class extending `ARCBaseGame` that implements game logic. The class name must match the 4-character game ID with the first letter capitalized.

## Directory Structure

To start building your own environment, create the following directory structure:
> 
```webp
ARC-AGI/
└── environment_files/
    └── ab12/               # use any game ID you want
        └── v1/             # use any version identifier you want
            ├── ab12.py     # must match game ID above
            └── metadata.json
```

> ### Metadata File

Create `metadata.json`:

```json theme={null}
{
  "game_id": "ab12-v1",
  "local_dir": "environment_files\\ab12\\v1"
}
```

| Field              | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `game_id`          | Unique identifier in format `{4-char game ID}-{version}` |
| `default_fps`      | Frames per second for playback (optional)                |
| `baseline_actions` | Array of average action counts per level (optional)      |
| `tags`             | Optional tags for categorization                         |
| `local_dir`        | Relative path to game directory                          |

> ## Imports and Constants

Start with the imports and constants:

```python theme={null}
import random

from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)

BACKGROUND_COLOR = 0  
PADDING_COLOR = 4     

# Available colors for sprites (excluding all black -> white colors for visibility)
SPRITE_COLORS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Parameters for Procedural Generation
MIN_SIZE = 1
MAX_SIZE = 4
```

> ## Sprite and Level Definitions

Define the sprites and levels for your game.\
In this example, the base sprite template that will be cloned and modified during generation:

```python theme={null}
# Base sprite template - single pixel will be cloned, scaled, and recolored
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
```

> ## Game Class Definition

The game class extends `ARCBaseGame` and accepts a seed parameter:

```python theme={null}
class Ab12(ARCBaseGame):
    """Click-to-remove game with seeded random sprite generation."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
    
        # Create camera with background and padding colors
        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            width=8,
            height=8,
        )

        # Initialize base game
        super().__init__(
            game_id="ab12",
            levels=levels,
            camera=camera,
        )
```

> ## Gameplay Logic and Mechanics
This game generates sprites by cloning from the sprite dictionary and applying random properties:

```python theme={null}
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
```

### Generation Pattern

The method chains operations on the cloned sprite:

1. **`.clone()`** - Creates independent copy of template
2. **`.color_remap(None, color)`** - Changes all pixels to random color
3. **`.set_scale(scale)`** - Applies the random scale
4. **`.set_position(x, y)`** - Places at random position in the grid

These can be used in level definitions directly for static levels.

>
> ## Level Initialization

The `on_set_level()` method is called when a level loads:

```python theme={null}
    def on_set_level(self, level: Level) -> None:
        """Called when the level is set."""
        # Generate sprites based on seed
        self.generate_sprites()
```

This method triggers sprite generation each time the level is set or reset.

> ## Win Condition

Define when the player wins:

```python theme={null}
    def _check_win(self) -> bool:
        """Check if all targets have been removed."""
        return len(self.current_level._sprites) == 0
```

## Game Loop

The `step()` method contains the main game logic.

### Action Flow

1. **Check action type**: `GameAction.ACTION6` is the click action
2. **Get coordinates**: Extract `x`, `y` from `action.data`
3. **Convert coordinates**: `display_to_grid()` handles camera scaling
4. **Find sprite**: Check if click hit any sprite `QUANT`
5. **Remove sprite**: Update level state
6. **Check win**: Advance to the next level or win the game on the last level if condition is met
7. **Complete action**: Always call `self.complete_action()`

- [quant]
```python theme={null}
    def step(self) -> None:
        """Process game logic for each step."""

        # 1. Check action type - GameAction.ACTION6 is the click action
        if self.action.id == GameAction.ACTION6:
            # 2. Get coordinates - Extract x, y from action.data
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)

            # 3. Convert coordinates - display_to_grid() handles camera scaling
            coords = self.camera.display_to_grid(x, y)

            if coords:
                grid_x, grid_y = coords

                # 4. Find sprite - Check if click hit any sprite
                clicked_sprite = self.current_level.get_sprite_at(grid_x, grid_y)
                if clicked_sprite:
                    # 5. Remove sprite - Update level state
                    self.current_level.remove_sprite(clicked_sprite)

                    # 6. Check win - Advance to next level if condition is met
                    if self._check_win():
                        self.next_level()

        # 7. Complete action - Always call self.complete_action()
        self.complete_action()
```

> ## Testing the Game

Run the game using the ARC-AGI-3 client:

```python theme={null}
import arc_agi
from arcengine import GameAction

# Default: looks for games in "environment_files" directory
arc = arc_agi.Arcade()
env = arc.make("ab12-v1", seed=0, render_mode="terminal")

# Or specify a custom directory
arc = arc_agi.Arcade(environments_dir="./my_games")
env = arc.make("ab12-v1", seed=0, render_mode="terminal")

# Perform clicks (ACTION6 with x, y coordinates)
env.step(GameAction.ACTION6, data={"x": 32, "y": 32})
```

If you'd like to see a 100-step run on your game, play the game with the [Sample Agent](./toolkit/minimal).

> ## Complete Code

<Accordion title="Full ab12.py source code">
 
  ```python theme={null}
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
  ```
</Accordion>

## Further Reading

For more information, refer to the **ARC Engine Documentation**.
