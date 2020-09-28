from typing import Union, List, Sequence, Tuple
from collections import namedtuple
import numpy as np

Vector = Union[np.ndarray, Sequence[float]]

Node = Union[str, int]
NodeList = List[Node]

Origin = str
Destination = str
Index = int

# Origin, Destination, Index
# example: lane_index1 = LaneIndex('a', 'b', 0)
LaneIndex = namedtuple('LaneIndex', 'origin destination index')


#LaneIndex = Tuple[str, str, int]

Route = List[LaneIndex]

#Route = List["AbstractLane"]

Path = List[Destination]

LanesList = List["AbstractLane"]

AngleRadians = float

AngleDegrees = float

class LineType:
    """
        A lane side line type.
    """
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


CL, SL, UL, NL = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.CONTINUOUS, LineType.NONE
