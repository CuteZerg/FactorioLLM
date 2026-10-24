"""
Synthetic User Prompts for this blueprint:
- Create a compact 4x10 belt and splitter balancing manifold with underground belts in Factorio using draftsman.
- Сделай компактный балансировщик 4 на 10 с подземными конвейерами и сплиттерами для Factorio.
- Generate a python blueprint script for a symmetrical 4-column belt routing setup featuring splitters and undergrounds.
"""

from draftsman.blueprintable import Blueprint
from draftsman.entity import TransportBelt, Splitter, UndergroundBelt

# Blueprint: Normalized Blueprint
bp = Blueprint()

for x in range(4):
    for y in range(10):
        # Determine entity type and direction based on position and column
        if y in [0, 9]:
            bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': float(y)}, direction=0))
        elif x in [0, 3]:
            if y == 1:
                bp.entities.append(Splitter('splitter', position={'x': float(x) + 0.5, 'y': float(y)}, direction=0))
            elif y == 2:
                bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': float(y)}, direction=0 if x == 0 else 0))
            elif y == 3:
                bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': float(y)}, direction=0 if x == 0 else 0))
            elif y == 5:
                bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': float(y)}, direction=4 if x == 0 else 12))
            elif y == 6:
                bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': float(y)}, direction=0))
            elif y == 7:
                bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': float(y)}, direction=0 if x == 0 else 0))
            elif y == 8:
                bp.entities.append(Splitter('splitter', position={'x': float(x) + 0.5, 'y': float(y)}, direction=0))
        elif x in [1, 2]:
            if y == 2:
                bp.entities.append(UndergroundBelt('underground-belt', position={'x': float(x), 'y': float(y)}, direction=0))
            elif y == 3:
                bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': float(y)}, direction=12 if x == 1 else 4))
            elif y == 5:
                bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': float(y)}, direction=0))
            elif y == 6:
                bp.entities.append(UndergroundBelt('underground-belt', position={'x': float(x), 'y': float(y)}, direction=0))
            elif y in [4, 7]:
                bp.entities.append(Splitter('splitter', position={'x': float(x) + 0.5, 'y': float(y)}, direction=0))

print(bp.to_string())