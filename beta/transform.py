import numpy as np
import pyrosetta

from pyrosetta.rosetta.numeric import xyzVector_double_t as V3

def transform_pose(p: pyrosetta.Pose, xform: np.array) -> None:
	coords = []

	for res in p.residues:
		for atom in res.atoms():
			c = np.array([list(atom.xyz()) + [1.0]], dtype=np.float)
			coords.append(c)
	
	index = 0
	transformed_coords = np.dot(xform, np.stack(coords).T)

	for res in p.residues:
		for atom in res.atoms():
			atom.xyz(V3(*transformed_coords[:3, index]))
			index += 1

