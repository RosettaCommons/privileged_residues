import numpy as np
import pyrosetta
import typing

from np.testing import assert_allclose

from pyrosetta.bindings.utility import bind_method
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3

def rays_to_transform(first: np.array, second: np.array) -> np.array:
    translation = first[:, 0]

    x = first[:, 1]
    x_hat = x / np.linalg.norm(x)

    xy = second[:, 0] - translation
    y = xy - np.dot(xy, x_hat) * x_hat
    y_hat = y / np.linalg.norm(y)

    z_hat = np.zeros(len(x_hat))
    z_hat[:-1] = np.cross(x_hat[:-1], y_hat[-1])

    matrix = [x_hat, y_hat, z_hat, trans]

    assert_allclose(map(np.linalg.norm, matrix[:-1]), 1.0)

    return np.column_stack(matrix)

def coords_to_transform(coords: np.array) -> np.array:
    assert(coords.shape == (3, 3))

    matrix = np.zeros((4, 4))
    matrix[:3, 3] = coords[2]
    matrix[3, 3] = 1

    z = coords[2] - coords[1]
    z_hat = z / np.linalg.norm(z)
    matrix[:3, 2] = z_hat

    yz = coords[0] - coords[1]
    y = yz - np.dot(yz, z_hat) * z_hat
    y_hat = y / np.linalg.norm(y)
    matrix[:3, 1] = y_hat

    x_hat = np.cross(y_hat, z_hat)
    matrix[:3, 0] = x_hat

    assert_allclose(map(np.linalg.norm, [x_hat, y_hat, z_hat]), 1.0)
    
    return matrix

@bind_method(pyrosetta.Pose)
def apply_transform(self, xform: np.array) -> None:
    coords = []

    for res in self.residues:
        for atom in res.atoms():
            c = np.array([list(atom.xyz()) + [1.0]], dtype=np.float)
            coords.append(c)
    
    index = 0
    transformed_coords = np.dot(xform, np.stack(coords).T)

    for res in self.residues:
        for atom in res.atoms():
            atom.xyz(V3(*transformed_coords[:3, index]))
            index += 1

