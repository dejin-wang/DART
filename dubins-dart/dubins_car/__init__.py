"""Dubins-car benchmark for direction-adaptive skill objectives."""

from dubins_car.env import DubinsCarEnv, register_dubins_car

register_dubins_car()

__all__ = ["DubinsCarEnv", "register_dubins_car"]
