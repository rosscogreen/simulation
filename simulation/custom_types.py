from typing import Union, List, Sequence, Tuple
from collections import namedtuple
import numpy as np

Vector = Union[np.ndarray, Sequence[float]]

Origin = str
Destination = str
Index = int

# Origin, Destination, Index
#LaneIndex = namedtuple('Lane', 'o d i')
LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]
Path = List[Destination]

class LineType:
    """
        A lane side line type.
    """
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


CL, SL, UL, NL = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.CONTINUOUS, LineType.NONE
