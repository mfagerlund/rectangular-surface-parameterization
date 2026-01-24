"""Grid vertex data structures for QEx quad extraction.

This module defines the core data structures used to represent the integer
grid vertices and edges that form the basis of quad mesh extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class GridVertex:
    """A vertex on the integer UV grid.

    Grid vertices are points where integer UV coordinates fall on the mesh.
    They can be located:
    - Inside a triangle face ('face')
    - On a triangle edge ('edge')
    - At a mesh vertex ('vertex')

    Each grid vertex has up to 4 outgoing edge directions corresponding to
    the cardinal directions in UV space: +u, +v, -u, -v.

    Attributes:
        type: Location type - 'face', 'edge', or 'vertex'
        uv: Integer UV coordinates as (u, v) tuple
        pos_3d: 3D position on the mesh surface, shape (3,)
        face_idx: Index of the triangle containing this vertex
        edge_idx: For 'edge' type, the mesh edge index (None otherwise)
        vertex_idx: For 'vertex' type, the mesh vertex index (None otherwise)
        outgoing_edges: Dict mapping direction (0,1,2,3) to target info.
            Direction encoding: 0=+u, 1=+v, 2=-u, 3=-v
            Each value is a tuple of (target_vertex_idx, reverse_direction)
            or None if no connection in that direction.
    """

    type: str
    uv: Tuple[int, int]
    pos_3d: np.ndarray
    face_idx: int
    edge_idx: Optional[int] = None
    vertex_idx: Optional[int] = None
    outgoing_edges: Dict[int, Optional[Tuple[int, int]]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the grid vertex and initialize outgoing edges."""
        if self.type not in ('face', 'edge', 'vertex'):
            raise ValueError(f"Invalid type: {self.type}. Must be 'face', 'edge', or 'vertex'")

        if not isinstance(self.uv, tuple) or len(self.uv) != 2:
            raise ValueError(f"uv must be a tuple of 2 integers, got {self.uv}")

        if self.pos_3d.shape != (3,):
            raise ValueError(f"pos_3d must have shape (3,), got {self.pos_3d.shape}")

        # Initialize all 4 directions to None if not provided
        for direction in range(4):
            if direction not in self.outgoing_edges:
                self.outgoing_edges[direction] = None

    def has_connection(self, direction: int) -> bool:
        """Check if there's a connection in the given direction.

        Args:
            direction: 0=+u, 1=+v, 2=-u, 3=-v

        Returns:
            True if there's a valid connection, False otherwise
        """
        return self.outgoing_edges.get(direction) is not None

    def get_target(self, direction: int) -> Optional[Tuple[int, int]]:
        """Get the target vertex and reverse direction for a given direction.

        Args:
            direction: 0=+u, 1=+v, 2=-u, 3=-v

        Returns:
            Tuple of (target_vertex_idx, reverse_direction) or None
        """
        return self.outgoing_edges.get(direction)


@dataclass
class GridEdge:
    """An edge connecting two grid vertices.

    Grid edges represent connections between grid vertices along the
    cardinal directions in UV space.

    Attributes:
        from_vertex: Index of the source GridVertex in the vertex list
        to_vertex: Index of the target GridVertex (-1 if unconnected)
        direction: Direction from source - 0=+u, 1=+v, 2=-u, 3=-v
        reverse_direction: Direction at target vertex pointing back to source
            (opposite of how we arrived at the target)
    """

    from_vertex: int
    to_vertex: int
    direction: int
    reverse_direction: int

    def __post_init__(self):
        """Validate the grid edge."""
        if not 0 <= self.direction <= 3:
            raise ValueError(f"direction must be 0-3, got {self.direction}")
        if not 0 <= self.reverse_direction <= 3:
            raise ValueError(f"reverse_direction must be 0-3, got {self.reverse_direction}")

    @property
    def is_connected(self) -> bool:
        """Check if this edge has a valid target."""
        return self.to_vertex >= 0


# Direction constants for clarity
DIRECTION_PLUS_U = 0
DIRECTION_PLUS_V = 1
DIRECTION_MINUS_U = 2
DIRECTION_MINUS_V = 3


def opposite_direction(direction: int) -> int:
    """Get the opposite direction.

    Args:
        direction: 0=+u, 1=+v, 2=-u, 3=-v

    Returns:
        Opposite direction (0<->2, 1<->3)
    """
    return (direction + 2) % 4


def direction_to_uv_delta(direction: int) -> Tuple[int, int]:
    """Convert a direction to UV delta.

    Args:
        direction: 0=+u, 1=+v, 2=-u, 3=-v

    Returns:
        Tuple of (du, dv) for unit step in that direction
    """
    deltas = {
        DIRECTION_PLUS_U: (1, 0),
        DIRECTION_PLUS_V: (0, 1),
        DIRECTION_MINUS_U: (-1, 0),
        DIRECTION_MINUS_V: (0, -1),
    }
    return deltas[direction]


def uv_delta_to_direction(du: int, dv: int) -> Optional[int]:
    """Convert a UV delta to direction.

    Args:
        du: Change in u coordinate (sign only matters)
        dv: Change in v coordinate (sign only matters)

    Returns:
        Direction (0-3) or None if not a cardinal direction
    """
    if du > 0 and dv == 0:
        return DIRECTION_PLUS_U
    elif du < 0 and dv == 0:
        return DIRECTION_MINUS_U
    elif du == 0 and dv > 0:
        return DIRECTION_PLUS_V
    elif du == 0 and dv < 0:
        return DIRECTION_MINUS_V
    return None
