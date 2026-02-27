"""Frozen interface and shared constants for Asteroid Belt Prospector."""

N_COMMODITIES = 6
COMMODITY_NAMES = ("iron", "nickel", "water_ice", "pge", "rare_isotopes", "volatiles")

MAX_NODES = 32
MAX_NEIGHBORS = 6
MAX_ASTEROIDS = 16

NODE_TYPES = 3
NODE_STATION = 0
NODE_CLUSTER = 1
NODE_HAZARD = 2

OBS_DIM = 260
N_ACTIONS = 69

CREDIT_SCALE = 1000.0
FUEL_MAX = 1000.0
HULL_MAX = 100.0
HEAT_MAX = 100.0
TOOL_MAX = 100.0
CARGO_MAX = 200.0
ALERT_MAX = 100.0
TIME_MAX = 20000.0

INVALID_ACTION_PENALTY = -0.01
