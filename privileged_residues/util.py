import numpy as np
import pyrosetta

from numpy.testing import assert_allclose

from pyrosetta.bindings.utility import bind_method
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3
from pyrosetta.toolbox.numpy_utils import numpy_to_rosetta

from rif.geom import Ray

@bind_method(pyrosetta.Pose)
def apply_transform(self, xform):
    """Apply a homogeneous transform to the current pose.

    Parameters
    ----------
    xform : np.ndarray
        A homogeneous transform.
    """

    assert(xform.shape == (4, 4)) # homogeneous transform
    assert_allclose(np.linalg.det(xform[:3,:3]), 1., atol=1e-4)

    Rx = numpy_to_rosetta(xform[:3, :3])
    v = V3(*xform[:3, 3])

    self.apply_transform_Rx_plus_v(Rx, v)


@bind_method(pyrosetta.rosetta.numeric.xyzVector_double_t)
def __iter__(self):
    """Generator for pyrosetta.rosetta.numeric.xyzVector_double_t
    instances. Makes casting directly to numpy.array possible.

    Yields
    ------
    float
        The next coordinate value in the vector in the order x, y, z.

    Examples
    --------
    >>> print([i for i in pose.residues[1].xyz(1)])
    [13.092, 4.473, -2.599]

    >>> np.array([*pose.residues[1].xyz(1)])
    array([ 13.092,   4.473,  -2.599])
    """

    for value in [self.x, self.y, self.z]:
        yield value

def numpy_to_rif(r):
    """Convert from NumPy ray representation to RIF ray representation.

    Parameters
    ----------
    r : np.ndarray
        Input NumPy ray.

    Returns
    -------
    rif.geom.Ray
    """
    return r.astype("f4").reshape(r.shape[:-2] + (8,)).view(Ray)

def models_from_pdb(fname):
    """Get models from a PDB as individual poses.

    Parameters
    ----------
    fname : str
        Path to a PDB.

    Yields
    ------
    pyrosetta.Pose
        The next model in the PDB.
    """

    p = pyrosetta.Pose()

    with open(fname, "r") as f:
        model = []

        for l in f:
            if (l.startswith("#")):
                continue

            line = l.rstrip()
            model.append(line)

            if (line.startswith("ENDMDL")):
                pose_from_pdbstring(p, pdbcontents="\n".join(model))
                yield p.clone()
                model = []

        if (len(model)):
            pose_from_pdbstring(p, pdbcontents="\n".join(model))
            yield p.clone()

