import numpy as np
import pyrosetta
import typing

from pyrosetta.bindings.utility import bind_method
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3

@bind_method(pyrosetta.Pose)
def apply_transform(self, xform: np.array) -> None:
    coords = []

    for res in self.residues:
        for atom in res.atoms():
            c = np.array(list(atom.xyz()) + [1.0], dtype=np.float)
            coords.append(c)

    index = 0
    transformed_coords = np.dot(xform, np.stack(coords).T)

    for res in self.residues:
        for atom in res.atoms():
            atom.xyz(V3(*transformed_coords[:3, index]))
            index += 1

@bind_method(pyrosetta.rosetta.numeric.xyzVector_double_t)
def __iter__(self):
	"""Generator for pyrosetta.rosetta.numeric.xyzVector_double_t
	instances. Makes casting directly to numpy.array possible.

	Yields:
		float: The next coordinate value in the vector in the order
			x, y, z.

	Examples:
		>>> print([i for i in pose.residues[1].xyz(1)])
		[13.092, 4.473, -2.599]

		>>> np.array([*pose.residues[1].xyz(1)])
		array([ 13.092,   4.473,  -2.599])
	"""
	for value in [self.x, self.y, self.z]:
		yield value

