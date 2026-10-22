"""Exact optimisation for agents. As a library:

    from optimizer import solve, get_format
    solve("assignment", {"workers": ["Ana", "Ben"], "jobs": ["J1", "J2"], "cost": [[4, 7], [6, 3]]})
"""
from .kinds import KIND_NAMES, catalogue, get_format, solve

__all__ = ["KIND_NAMES", "catalogue", "get_format", "solve"]
