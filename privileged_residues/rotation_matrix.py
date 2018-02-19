import numpy as np


def _is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def rot_ein(axis, angle, degrees='auto', dtype='f8', shape=(4, 4),
            center=None):
    """Compute the rotation matrix that describes the rotation by an
    angle about an axis and return it.

    Notes:
        The matrix for a rotation by an angle of theta about an axis in
        the direction of u can be written as
        ------------------------------------------------------------
        R = cos(theta) I + sin(theta) u_x + (1 - cos(theta)) (u X u)
        ------------------------------------------------------------
        1. 'I' is the identity matrix,
        2. 'u_x' is the cross product matrix of the unit vector
           representing the axis. See: https://en.wikipedia.org/wiki/\
           Cross_product#Conversion_to_matrix_multiplication; and
        3. '(u X u)' is the tensor product of u and itself. Since u can
           be written as a vector, the tensor product is equal to the
           Kronecker product. See: https://en.wikipedia.org/wiki/\
           Kronecker_product

        The function name comes from the fact that Einstein summation
        is used to compute the cross product matrix of the unit vector
        in the direction of the axis.

    Args:
        axis (numpy.array): An axis or a vector of axes to rotate about.
        angle (numpy.array OR float): The angle or a vector angles by
            which to rotate.
        degrees (str): Set to True to indicate that the value being
            passed as `angle` is in degrees, False or None or indicates
            the value is in radians. Setting degrees to 'auto' will
            cause the script to make its best guess to determine
            whether the value is in degrees or radians. Defaults to
            'auto'.
        dtype (str): The dtype for the values in the matrix. Defaults
            to 'f8'.
        shape (tuple): Shape of the output rotation matrix. Defaults to
            (3, 3).

    Returns:
        numpy.array: Rotation matrix describing a rotation of `angle`
        about `axis` in the specified shape.
    """
    axis = np.array(axis, dtype=dtype)
    angle = np.array(angle, dtype=dtype)
    if degrees is 'auto':
        degrees = guess_is_degrees(angle)
    angle = angle * np.pi / 180.0 if degrees else angle
    if axis.shape and angle.shape and not _is_broadcastable(
            axis.shape[:-1], angle.shape):
        raise ValueError('axis and angle not compatible: ' +
                         str(axis.shape) + ' ' + str(angle.shape))
    axis /= np.linalg.norm(axis, axis=-1)[..., np.newaxis]

    center = (np.array([0, 0, 0], dtype=dtype) if center is None
              else np.array(center, dtype=dtype))

    outshape = angle.shape if angle.shape else axis.shape[:-1]
    r = np.zeros(outshape + shape, dtype=dtype)

    sin_vec = np.sin(angle)[..., np.newaxis, np.newaxis]
    cos_vec = np.cos(angle)[..., np.newaxis, np.newaxis]

    # Define the Levi-Civita tensor in R3
    # https://en.wikipedia.org/wiki/Levi-Civita_symbol#Three_dimensions
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    axis_cross_prod_mat = np.einsum('ijk,...j->...ik', eijk, axis[..., :3])

    kronecker_prod = axis[..., np.newaxis, :3] * axis[..., :3, np.newaxis]

    r[..., :3, :3] = cos_vec * np.identity(3) + \
        sin_vec * axis_cross_prod_mat + \
        (1 - cos_vec) * kronecker_prod

    x, y, z = center[..., 0], center[..., 1], center[..., 2]
    r[..., 0, 3] = x - r[..., 0, 0] * x - r[..., 0, 1] * y - r[..., 0, 2] * z
    r[..., 1, 3] = y - r[..., 1, 0] * x - r[..., 1, 1] * y - r[..., 1, 2] * z
    r[..., 2, 3] = z - r[..., 2, 0] * x - r[..., 2, 1] * y - r[..., 2, 2] * z
    r[..., 3, 3] = 1
    return r
