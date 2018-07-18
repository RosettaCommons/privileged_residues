import numpy as np
import pyrosetta

from numpy.testing import assert_allclose

def rays_to_transform(first, second):
    """From two rays, construct a homogeneous transform.

    Parameters
    ----------
    first : np.ndarray
    second : np.ndarray

    Returns
    -------
    np.ndarray
        Homogeneous transform constructed from the input rays.
    """

    translation = first[:, 0]

    x = first[:, 1]
    x_hat = x / np.linalg.norm(x)

    xy = second[:, 0] - translation
    y = xy - np.dot(xy, x_hat) * x_hat
    y_hat = y / np.linalg.norm(y)

    z_hat = np.zeros(len(x_hat))
    z_hat[:-1] = np.cross(x_hat[:-1], y_hat[:-1])
    assert_allclose(np.linalg.norm(z_hat), 1.0)

    matrix = [x_hat, y_hat, z_hat, translation]

    assert_allclose(list(map(np.linalg.norm, matrix[:-1])), 1.0)

    return np.column_stack(matrix)

def coords_to_transform(coords):
    """From a set of coordinates, construct a homogeneous transform.

    Parameters
    ----------
    coords : np.ndarray

    Returns
    -------
    np.ndarray
        Homogeneous transform constructed from the input coordinates.
    """

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

    x_hat = np.cross(y_hat[:-1], z_hat[:-1])
    matrix[:3, 0] = x_hat

    assert_allclose(list(map(np.linalg.norm, [x_hat, y_hat, z_hat])), 1.0)
    
    return matrix

def create_ray(center, base):
    """Create a ray of unit length from two points in space and return
    it.

    Notes
    -----
    The ray is constructed such that:

    1. The direction points from `base` to `center` and is unit length.
    2. The point at which the ray is centered is at `center`.

    Parameters
    ----------
    center : numpy.ndarray
        A (1, 3) array representing the coordinate at which to center
        the resulting ray.
    base : numpy.ndarray
        A (1, 3) array representing the base used to determine the
        direction of the resulting ray.

    Returns
    -------
    numpy.ndarray
        A (2,4) array representing a ray in space with a point and a
        unit direction.
    """
    direction = center - base
    direction /= np.linalg.norm(direction)

    ray = np.empty((4, 2))
    ray[:-1, 0] = center
    ray[-1, 0] = 1.  # point!

    ray[:-1, 1] = direction
    ray[-1, 1] = 0.  # direction!

    return ray

