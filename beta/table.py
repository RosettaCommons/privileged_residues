import numpy as np
import rif

from collections import namedtuple

from rif.geom import rif.geom.Ray
from rif.geom.ray_hash import RayToRay4dHash
from rif.hash import XformHash_bt24_BCC6_X3f

from typing import Iterable, MappingView

import .transform

# NOTE(onalant): <C-g> will show current file in nvi!

_LEVER = 10
_BOUND = 1000

class ResidueTable(Mapping[np.uint64, Iterable[pyrosetta.Pose]]):

	def __init__(self, cart_resl: float = 0.1, ori_resl: float = 2.0, cart_bound: float = 16.0) -> None:
		self._raygrid = RayToRay4dHash(ori_resl, _LEVER, bound=_BOUND)
		self._lattice = XformHash_bt24_BCC6_X3f(cart_resl, ori_resl, cart_bound)
		self._table = None

	def __get_item__(self, key: np.uint64) -> Iterable[pyrosetta.Pose]:
		transform = self._lattice.get_center([bin])['raw'].squeeze()
		results = self.table[key]

	def __iter__(self):
		pass

	def __len__(self):
		return len(self.table)

