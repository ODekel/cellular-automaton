import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from map_builder import ImageConfig, build_map
from weather_viewer import WeatherSimulatorViewer

GridType = npt.NDArray[np.float32]
GridWindsType = npt.NDArray[np.int8]
GridTilesType = npt.NDArray[np.uint8]
GridCloudsType = npt.NDArray[np.bool]


class TileType(Enum):
    Water = np.uint8(0)
    Ice = np.uint8(1)
    Desert = np.uint8(2)
    Grassland = np.uint8(3)
    Forest = np.uint8(4)
    City = np.uint8(5)


WIND_DIRECTIONS = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]], dtype=np.int8)


@dataclass
class Grid:
    tiles: GridTilesType
    temps: GridType
    winds: GridWindsType
    pollution: GridType
    clouds: GridCloudsType

    def __post_init__(self):
        if self.tiles.shape != self.temps.shape or self.tiles.shape != self.pollution.shape or\
           self.tiles.shape != self.clouds.shape or\
           self.tiles.shape != self.winds.shape[:-1] or self.winds.shape[-1] != 2:
            raise ValueError("Grids must have matching shapes")


@dataclass
class SimulatorConfig:
    forest_pollution_absorption_rate: np.float32
    ice_heat_dissipation_rate: np.float32
    desert_heat_absorption_rate: np.float32
    city_pollution_generation_rate: np.float32
    heat_spread_multiplier: np.float32
    clouds_form_probability: np.float32
    clouds_rain_probability: np.float32
    ice_threshold: np.float32


class WeatherSimulator:
    def __init__(self, grid: Grid, config: SimulatorConfig):
        self._grid = grid
        self._indices = np.indices(grid.tiles.shape).transpose((1, 2, 0))
        self._config = config
        self._rng = np.random.default_rng()

    @property
    def grid(self) -> Grid:
        return self._grid

    def next(self):
        h, w = self._grid.tiles.shape
        winds_loc = (self._indices + self._grid.winds).astype(np.intp)
        winds_loc[..., 0] %= h
        winds_loc[..., 1] %= w
        winds_loc_r = winds_loc[..., 0]
        winds_loc_c = winds_loc[..., 1]

        self._next_temps(winds_loc_r, winds_loc_c)
        self._next_pollution(winds_loc_r, winds_loc_c)
        self._next_clouds(winds_loc_r, winds_loc_c)
        self._next_winds()
        self._next_tiles()


    def _next_temps(self, winds_loc_r, winds_loc_c):
        temps_diff = (self._grid.temps[winds_loc_r, winds_loc_c] - self._grid.temps) * self._config.heat_spread_multiplier
        self._grid.temps += temps_diff + self._grid.pollution
        np.add.at(self._grid.temps, (winds_loc_r, winds_loc_c), -temps_diff)
        self._grid.temps[self._grid.tiles == TileType.Ice.value] -= self._config.ice_heat_dissipation_rate
        self._grid.temps[self._grid.tiles == TileType.Desert.value] += self._config.desert_heat_absorption_rate

    def _next_pollution(self, winds_loc_r, winds_loc_c):
        forests = self._grid.tiles == TileType.Forest.value
        next_pollution = self._grid.pollution.copy()
        next_pollution[forests] = \
            np.maximum(self._grid.pollution[forests] - self._config.forest_pollution_absorption_rate,
                       np.zeros_like(next_pollution[forests]))
        next_pollution /= 2
        np.add.at(next_pollution, (winds_loc_r, winds_loc_c), next_pollution)
        next_pollution[self._grid.tiles == TileType.City.value] += self._config.city_pollution_generation_rate
        self._grid.pollution = next_pollution.astype(np.float32)

    def _next_clouds(self, winds_loc_r, winds_loc_c):
        next_clouds = np.zeros_like(self._grid.clouds).astype(np.bool)
        np.add.at(next_clouds, (winds_loc_r, winds_loc_c), self._grid.clouds)
        clouds_mask = next_clouds == True
        next_clouds[clouds_mask] = self._rng.random(next_clouds[clouds_mask].shape) >= self._config.clouds_rain_probability
        no_clouds_mask = next_clouds == False
        next_clouds[no_clouds_mask] = self._rng.random(next_clouds[no_clouds_mask].shape) <= self._config.clouds_form_probability
        self._grid.clouds = next_clouds

    def _next_winds(self):
        self._grid.winds = self._rng.choice(WIND_DIRECTIONS, self._grid.tiles.shape)

    def _next_tiles(self):
        self._grid.tiles[(self._grid.tiles == TileType.Ice.value) & (self._grid.temps > self._config.ice_threshold)] = TileType.Water.value
        self._grid.tiles[(self._grid.tiles == TileType.Water.value) & (self._grid.temps < self._config.ice_threshold)] = TileType.Ice.value


config = json.load(open("config.json"))
colors = {}

temps_map = {TileType[name.capitalize()].value: value for name, value in config["temps"].items()}
tiles_to_temps = np.vectorize(lambda tile: temps_map[tile], otypes=[np.float32])

build_config = lambda d: {''.join(('_' + letter.lower() if letter.isupper() else letter for letter in key)): value for key, value in d.items()}
sim_config = SimulatorConfig(**build_config(config['simulation']))
image_config = ImageConfig(**build_config(config['image']), tiles_mapping={tile.name: tile.value for tile in TileType})
tiles = build_map(image_config)
temps = tiles_to_temps(tiles)
pollution = np.zeros_like(tiles, dtype=np.float32)
rng = np.random.default_rng()
winds = rng.choice(WIND_DIRECTIONS, tiles.shape)
clouds = rng.random(tiles.shape) <= (sim_config.clouds_form_probability / sim_config.clouds_rain_probability)
grid = Grid(tiles, temps, winds, pollution, clouds)

sim = WeatherSimulator(grid, sim_config)

viewer = WeatherSimulatorViewer(sim)
viewer.run()
