# this script will be how I load and search hash tables
import numpy as np

from collections import namedtuple
from os import path

# The following packages are not pip-installable
# The import calls are wrapped in a try/except block
try:
    import pyrosetta
except ImportError:
    print('Module "pyrosetta" not found in the current environment! '
          'Go to http://www.pyrosetta.org to download it.')
    pass

try:
    from rif.hash import *
except ImportError:
    print('Module "rif" not found in the current environment! '
          'Go to https://github.com/willsheffler/rif for more information.')
    pass

from . import hbond_ray_pairs, process_networks

HashTableData = namedtuple('HashTableData', ['type', 'cart_resl', 'ori_resl',
                                             'cart_bound'])
HashTableData.__doc__ = """Group the name of the type of interaction
and parameters to define a six-dimensional Body Centered Cube (BCC)
lattice used for hashing.

Attributes:
    type (str): Name of the types of ray pairs describing the
        interaction.
    cart_resl (float): The cartesian resolution of the BCC lattice
    ori_resl (float): The orientational resolution of the BCC
        lattice. In general, ori_resl should be roughly
        20 * cart_resl.
    cart_bound (float): The bounds of the lattice.
"""


def _fname_to_HTD(fname_string):
    """Convert the filename of a pickled hash table to a HashTableData
    instance.

    Notes:
        This assumes the file was created by `process_networks.py`.

    Args:
        fname_string (str): a string formatted as 'T_T_CR_OR_CB.pkl'.

    Returns:
        HashTableData: A nameedtuple populated with the information
        encoded in `fname_string`.
    """
    c = path.splitext(path.basename(fname_string))[0].split('_')
    assert(len(c) == 5)
    t = '_'.join(c[:2])
    return HashTableData(t, float(c[2]), float(c[3]), float(c[4]))


def get_ht_from_table(bin, cart_resl, ori_resl, cart_bound):
    """Return the homogenous transform from a six-dimensional Body
    Centered  Cubic (BCC) lattice based on the supplied parameters
    corresponding to the supplied bin number.

    Args:
        bin (numpy.uint64): The bin number of the lattice.
        cart_resl (float): The cartesian resolution of the BCC lattice
        ori_resl (float): The orientational resolution of the BCC
            lattice. In general, ori_resl should be roughly
            20 * cart_resl.
        cart_bound (float): The bounds of the lattice.

    Returns:
        numpy.array: A (4, 4) array representing the homogenous
        transform at the center of the bin.
    """
    # these are kind of magic numbers
    # is this thing expensive to instantiate? I could move stuff around to make
    # sure this only happens once
    xh = XformHash_bt24_BCC6_X3f(cart_resl, ori_resl, cart_bound)
    # to go from key -> bin center (~the xform I care about) use...
    return xh.get_center([bin])['raw'].squeeze()


def transform_pose(p, xform):
    """Transform the atomic coordinates of Pose by a specified
    homogenous tranform.

    Args:
        p (pyrosetta.Pose): A Pose instance to be transformed.
        xform (numpy.array): A (4, 4) array representing a homogenous
            transform.
    """
    coords = []
    for i in range(1, p.size() + 1):
        for j in range(1, p.residues[i].natoms() + 1):
            c = np.array([*p.residues[i].xyz(j)] + [1])
            coords.append(c)

    new_coords = np.dot(xform, np.stack(coords).T)
    from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
    tot_atoms = 0
    for i in range(1, p.size() + 1):
        for j in range(1, p.residues[i].natoms() + 1):
            x, y, z = tuple(c for c in new_coords[:3, tot_atoms])
            p.residues[i].atom(j).xyz(xyzVec(x, y, z))
            tot_atoms += 1


if __name__ == '__main__':
    print('Don\'t execute me, bruh.')
