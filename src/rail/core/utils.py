# This is a deprecative path for importing RAILDIR, it is only for the transition period
# to move RAILDIR and find_rail_file to utils.path_utils

from rail.utils.path_utils import find_rail_file
from rail.utils.path_utils import RAILDIR

__all__ = [
    'find_rail_file',
    'RAILDIR',
]
