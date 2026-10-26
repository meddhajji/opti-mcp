"""Exact optimisation for agents. As a library:

    from optimizer import solve, get_format
    solve("assignment", {"workers": ["Ana", "Ben"], "jobs": ["J1", "J2"], "cost": [[4, 7], [6, 3]]})

GUIDANCE is the usage guidance the MCP server sends to clients, ready for a system prompt.
"""
from .kinds import GUIDANCE, KIND_NAMES, catalogue, get_format, solve

__all__ = ["GUIDANCE", "KIND_NAMES", "catalogue", "get_format", "solve"]
