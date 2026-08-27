from draftsman.blueprintable import Blueprint
from draftsman.entity import new_entity

# Original Blueprint: Unnamed
bp = Blueprint()

ent_0 = new_entity('transport-belt', position={'x': 123.5, 'y': 9.5}, direction=0)
bp.entities.append(ent_0)

ent_1 = new_entity('transport-belt', position={'x': 123.5, 'y': 11.5}, direction=0)
bp.entities.append(ent_1)

ent_2 = new_entity('transport-belt', position={'x': 123.5, 'y': 12.5}, direction=0)
bp.entities.append(ent_2)

ent_3 = new_entity('transport-belt', position={'x': 123.5, 'y': 14.5}, direction=4)
bp.entities.append(ent_3)

ent_4 = new_entity('transport-belt', position={'x': 123.5, 'y': 15.5}, direction=0)
bp.entities.append(ent_4)

ent_5 = new_entity('transport-belt', position={'x': 123.5, 'y': 16.5}, direction=0)
bp.entities.append(ent_5)

ent_6 = new_entity('transport-belt', position={'x': 123.5, 'y': 18.5}, direction=0)
bp.entities.append(ent_6)

ent_7 = new_entity('splitter', position={'x': 124.0, 'y': 10.5}, direction=0)
bp.entities.append(ent_7)

ent_8 = new_entity('splitter', position={'x': 124.0, 'y': 17.5}, direction=0)
bp.entities.append(ent_8)

ent_9 = new_entity('transport-belt', position={'x': 124.5, 'y': 9.5}, direction=0)
bp.entities.append(ent_9)

ent_10 = new_entity('underground-belt', position={'x': 124.5, 'y': 11.5}, direction=0)
bp.entities.append(ent_10)

ent_11 = new_entity('transport-belt', position={'x': 124.5, 'y': 12.5}, direction=12)
bp.entities.append(ent_11)

ent_12 = new_entity('transport-belt', position={'x': 124.5, 'y': 14.5}, direction=0)
bp.entities.append(ent_12)

ent_13 = new_entity('underground-belt', position={'x': 124.5, 'y': 15.5}, direction=0)
bp.entities.append(ent_13)

ent_14 = new_entity('transport-belt', position={'x': 124.5, 'y': 18.5}, direction=0)
bp.entities.append(ent_14)

ent_15 = new_entity('splitter', position={'x': 125.0, 'y': 13.5}, direction=0)
bp.entities.append(ent_15)

ent_16 = new_entity('splitter', position={'x': 125.0, 'y': 16.5}, direction=0)
bp.entities.append(ent_16)

ent_17 = new_entity('transport-belt', position={'x': 125.5, 'y': 9.5}, direction=0)
bp.entities.append(ent_17)

ent_18 = new_entity('underground-belt', position={'x': 125.5, 'y': 11.5}, direction=0)
bp.entities.append(ent_18)

ent_19 = new_entity('transport-belt', position={'x': 125.5, 'y': 12.5}, direction=4)
bp.entities.append(ent_19)

ent_20 = new_entity('transport-belt', position={'x': 125.5, 'y': 14.5}, direction=0)
bp.entities.append(ent_20)

ent_21 = new_entity('underground-belt', position={'x': 125.5, 'y': 15.5}, direction=0)
bp.entities.append(ent_21)

ent_22 = new_entity('transport-belt', position={'x': 125.5, 'y': 18.5}, direction=0)
bp.entities.append(ent_22)

ent_23 = new_entity('splitter', position={'x': 126.0, 'y': 10.5}, direction=0)
bp.entities.append(ent_23)

ent_24 = new_entity('splitter', position={'x': 126.0, 'y': 17.5}, direction=0)
bp.entities.append(ent_24)

ent_25 = new_entity('transport-belt', position={'x': 126.5, 'y': 9.5}, direction=0)
bp.entities.append(ent_25)

ent_26 = new_entity('transport-belt', position={'x': 126.5, 'y': 11.5}, direction=0)
bp.entities.append(ent_26)

ent_27 = new_entity('transport-belt', position={'x': 126.5, 'y': 12.5}, direction=0)
bp.entities.append(ent_27)

ent_28 = new_entity('transport-belt', position={'x': 126.5, 'y': 14.5}, direction=12)
bp.entities.append(ent_28)

ent_29 = new_entity('transport-belt', position={'x': 126.5, 'y': 15.5}, direction=0)
bp.entities.append(ent_29)

ent_30 = new_entity('transport-belt', position={'x': 126.5, 'y': 16.5}, direction=0)
bp.entities.append(ent_30)

ent_31 = new_entity('transport-belt', position={'x': 126.5, 'y': 18.5}, direction=0)
bp.entities.append(ent_31)

# Check the result
print(bp.to_string())
