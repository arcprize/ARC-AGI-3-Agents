"""
Helpers for working with game frames.
"""

import base64
import json
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOR_PALETTE = {
    0: (0, 0, 0),  # Black
    2: (255, 0, 0),  # Red
    4: (0, 255, 0),  # Green
    5: (128, 128, 128),  # Gray
    6: (0, 0, 255),  # Blue
    7: (255, 255, 0),  # Yellow
    8: (255, 165, 0),  # Orange
    9: (128, 0, 128),  # Purple
    10: (255, 255, 255),  # White
    11: (128, 128, 128),  # Gray
    # Top of player - 12
}
SCALE_FACTOR = 15


def extract_rect_from_render(
    b64image: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    """
    Extract rectangle from base64-encoded PNG and return as base64.

    Useful for extracting the door and key for comparison.
    """
    # Decode base64 to PIL Image
    img_data = base64.b64decode(b64image)
    img = Image.open(BytesIO(img_data))

    # Crop the rectangle
    cropped = img.crop(
        (
            x * SCALE_FACTOR,
            y * SCALE_FACTOR,
            (x + width) * SCALE_FACTOR,
            (y + height) * SCALE_FACTOR,
        )
    )

    # Convert back to base64
    buffered = BytesIO()
    cropped.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def render_frame(
    array_3d: list[list[list[int]]], description: str, with_highlights: bool = True
) -> str:
    """
    Renders a game frame to a PNG image.
    """

    # Default color (white) if number not in palette
    default_color = (255, 255, 255)

    # Convert the 3D array to a NumPy array
    np_array = np.array(array_3d[0], dtype=np.uint8)

    # Original dimensions
    orig_height, orig_width = np_array.shape

    # Create an empty RGB image with scaled dimensions
    scaled_width = (orig_width + 1) * SCALE_FACTOR
    scaled_height = (orig_height + 1) * SCALE_FACTOR

    # Add extra height for description
    description_height = 40
    img = Image.new("RGB", (scaled_width, scaled_height + description_height))
    pixels = img.load()

    with open("frame.json", "w") as f:
        f.write(json.dumps(np_array.tolist()))

    # Fill the image with colors from the palette, scaled
    for y in range(orig_height):
        for x in range(orig_width):
            color_num = np_array[y, x]
            color = COLOR_PALETTE.get(color_num, default_color)
            for i in range(SCALE_FACTOR):
                for j in range(SCALE_FACTOR):
                    pixels[
                        (x * SCALE_FACTOR + j) + SCALE_FACTOR,
                        (y * SCALE_FACTOR + i) + SCALE_FACTOR,
                    ] = color

    draw = ImageDraw.Draw(img)
    line_color = (128, 128, 128)  # Gray

    # Horizontal lines
    for k_h in range(orig_height + 1):
        y = k_h * SCALE_FACTOR
        draw.line([(0, y), (scaled_width - 1, y)], fill=line_color, width=1)
    draw.line(
        [(0, scaled_height - 1), (scaled_width - 1, scaled_height - 1)],
        fill=line_color,
        width=1,
    )

    # Vertical lines
    for k_v in range(orig_width + 1):
        x = k_v * SCALE_FACTOR
        draw.line([(x, 0), (x, scaled_height - 1)], fill=line_color, width=1)
    draw.line(
        [(scaled_width - 1, 0), (scaled_width - 1, scaled_height - 1)],
        fill=line_color,
        width=1,
    )

    font = ImageFont.load_default()
    text_color = (255, 255, 255)

    # Draw column numbers
    y_center_col_labels = SCALE_FACTOR / 2.0
    for col_idx in range(orig_width):
        x_center_col_labels = (
            col_idx * SCALE_FACTOR + (SCALE_FACTOR / 2.0)
        ) + SCALE_FACTOR
        draw.text(
            (x_center_col_labels, y_center_col_labels),
            str(col_idx),
            font=font,
            fill=text_color,
            anchor="mm",
        )

    # Draw row numbers
    x_center_row_labels = SCALE_FACTOR / 2.0
    for row_idx in range(orig_height):
        y_center_row_labels = (
            row_idx * SCALE_FACTOR + (SCALE_FACTOR / 2.0)
        ) + SCALE_FACTOR
        draw.text(
            (x_center_row_labels, y_center_row_labels),
            str(row_idx),
            font=font,
            fill=text_color,
            anchor="mm",
        )

    # Draw highlights
    if with_highlights:
        add_highlight(
            draw,
            ((2, 2), (47, 5)),
            "Health",
        )
        add_highlight(
            draw,
            ((52, 1), (64, 5)),
            "Lives",
        )

        found_player = False
        found_door = False
        found_rotator = False
        for y in range(orig_height):
            if found_player and found_door and found_rotator:
                break
            for x in range(orig_width):
                if np_array[y, x] == 12 and not found_player:
                    found_player = True
                    add_highlight(
                        draw,
                        ((x + 1, y + 1), (x + 9, y + 9)),
                        "Player",
                    )
                    break
                if np_array[y, x] == 5 and not found_door:
                    found_door = True
                    add_highlight(
                        draw,
                        ((x + 1, y + 1), (x + 9, y + 9)),
                        "Door",
                    )
                    break
                if (
                    np_array[y, x] == 9
                    and np_array[y - 1, x] == 3
                    and not found_rotator
                ):
                    found_rotator = True
                    add_highlight(
                        draw,
                        ((x + 1, y + 1), (x + 9, y + 9)),
                        "Rotator",
                    )
                    break

        add_highlight(draw, ((2, 55), (11, 64)), "Key")

    # Add description text at the bottom
    description_y = scaled_height + (description_height / 2)
    draw.text(
        (scaled_width / 2, description_y),
        description,
        font=font,
        fill=text_color,
        anchor="mm",
    )

    # Convert image to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str


def add_highlight(
    draw: ImageDraw.ImageDraw,
    coords: ImageDraw.Coords,
    label: str,
) -> None:
    (x1, y1), (x2, y2) = coords

    draw.rectangle(
        (
            (x1 * SCALE_FACTOR + 1, y1 * SCALE_FACTOR + 1),
            (x2 * SCALE_FACTOR - 1, y2 * SCALE_FACTOR - 1),
        ),
        fill=None,
        outline=(255, 0, 0),
        width=4,
    )

    # Text label
    draw.rectangle(
        (
            (((x1 + x2) / 2 - 2) * SCALE_FACTOR, (y2 + 0.25) * SCALE_FACTOR),
            (((x1 + x2) / 2 + 2) * SCALE_FACTOR, (y2 + 2) * SCALE_FACTOR),
        ),
        fill=(0, 0, 0),
        outline=None,
    )
    draw.text(
        ((x1 + x2) / 2 * SCALE_FACTOR, (y2 + 1) * SCALE_FACTOR),
        label,
        font=ImageFont.load_default(size=16),
        fill=(255, 255, 255),
        anchor="mm",
    )
