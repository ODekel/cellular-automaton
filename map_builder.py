from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageColor


def _pixel_to_tile(pixel: npt.NDArray[np.uint8], colors: dict[tuple, str], tiles_mapping: dict[str, np.uint8]) -> np.uint8:
    return tiles_mapping[colors[tuple(pixel)]]


_pixels_to_tiles = np.vectorize(_pixel_to_tile, otypes=[np.uint8], excluded=[1, 2], signature='(3)->()')


# Convert the image to only use colors from config.
def quantize(image: Image.Image, palette: list[tuple[int, int, int]]) -> Image.Image:
    palette_image = Image.new('P', (1, 1))
    filled_palette = [color for pixel in palette for color in pixel]
    # Need to fill palette array to 768 items, this is just how the library works for RGB (256 * 3).
    filled_palette += (768 - len(filled_palette))*[0]
    palette_image.putpalette(filled_palette)
    return image.convert('RGB').quantize(palette=palette_image)


@dataclass
class ImageConfig:
    path: str
    encoding: dict[str, list[str]]
    tiles_mapping: dict[str, np.uint8]


def build_map(config: ImageConfig) -> npt.NDArray[np.uint8]:
    # Create dictionary to convert color values to tile names.
    colors = {}
    for name, values in config.encoding.items():
        capitalized = name.capitalize()
        for value in values:
            colors[ImageColor.getrgb(value)] = capitalized
    image = Image.open(config.path)
    return _pixels_to_tiles(np.array(quantize(image, list(colors.keys())).convert('RGB'), dtype=np.uint8), colors, config.tiles_mapping)
