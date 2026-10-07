import re
import draftsman.entity as _d_ent

def _compat_getattr(name):
    if name in _d_ent.__dict__:
        return _d_ent.__dict__[name]
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1-\2', name)
    kebab = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', s1).lower()
    try:
        return _d_ent.get_entity_class(kebab)
    except Exception:
        raise AttributeError(f"module 'draftsman.entity' has no attribute '{name}'")

_d_ent.__getattr__ = _compat_getattr

from draftsman.blueprintable import Blueprint
from draftsman.entity import TransportBelt, StoneFurnace

bp = Blueprint()
for x in range(10):
    bp.entities.append(TransportBelt("fast-transport-belt", position={"x": float(x), "y": 0.0}, direction=4))

for i in range(8):
    bp.entities.append(StoneFurnace("stone-furnace", position={"x": float(i * 3), "y": 3.0}))

print("Original entities count:", len(bp.entities))
for e in bp.entities[:3]:
    print("Entity dict:", e.to_dict())
