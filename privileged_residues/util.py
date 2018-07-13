import numpy as np
import pyrosetta

from pyrosetta.bindings.utility import bind_method
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3
from pyrosetta.toolbox.numpy_utils import numpy_to_rosetta

from rif.geom import Ray

@bind_method(pyrosetta.Pose)
def apply_transform(self, xform):
	""" description - note the type and shape of the arguments """

	assert(xform.shape == (4, 4)) # homogeneous transform

	M = numpy_to_rosetta(xform[:3, :3])
	v = V3(*xform[:3, 3])

	self.apply_transform_Rx_plus_v(M, v)


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

def numpy_to_rif(r):
    return r.astype("f4").reshape(r.shape[:-2] + (8,)).view(Ray)

