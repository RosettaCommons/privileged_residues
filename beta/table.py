import numpy as np

from collections import namedtuple
from rif.hash import XformHash_bt24_BCC6_X3f
from typing import Iterator, MappingView

class ResidueTable(Mapping[np.uint64, pyrosetta.Pose]):

	def __init__(self, cart_resl: float = 0.1, ori_resl: float = 2.0, cart_bound: float = 16.0) -> None:
		self._lattice = XformHash_bt24_BCC6_X3f(cart_resl, ori_resl, cart_bound)
		self._table = None

	def __index__(self, key: np.uint64) -> Iterator[pyrosetta.Pose]:
		transform = xh.get_center([bin])['raw'].squeeze()
		results = self.table[key]

	def _transform_pose(self, pose: pyrosetta.Pose, xform: np.array):
		pass

