# this script will be how I load and search hash tables
import argparse
import numpy as np
import pickle
import sys

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
            lattice. In general, `ori_resl` should be roughly
            `20 * cart_resl`.
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


def tranform_pose(p, xform):
    """Tranform the atomic coordinates of Pose by a specified
    homogenous tranform.

    Args:
        p (pyrosetta.Pose): A Pose instance to be transformed.
        xform (numpy.array): A (4, 4) array representing a homogenous
            transform.
    """
    coords = []
    for i in range(1, p.size() + 1):
        for j in range(1, p.residue(i).natoms() + 1):
            c = np.ones(4)
            c[:3] = np.array([*p.residue(i).xyz(j)])
            coords.append(c)

    new_coords = np.dot(xform, np.stack(coords).T)
    from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
    tot_atoms = 0
    for i in range(1, p.size() + 1):
        for j in range(1, p.residue(i).natoms() + 1):
            x, y, z = tuple(c for c in new_coords[:3, tot_atoms])
            p.residue(i).atom(j).xyz(xyzVec(x, y, z))
            tot_atoms += 1


def main(argv):
    """Load the hash tables (pickled dictionaries) containing hydrogen
    bond ray pair hashes as keys and a functional group identifier and
    a bin number of Body Centered  Cubic (BCC) lattice corresponding to
    a homogenous transform that will place the functional group relative
    to the hydrogen bonding rays as values. In its current state, this
    script selects an arbitrary item from the table and ensures that a
    functional group can be placed correctly.
    """
    # setup per-use variables with argparse
    dir = 'functional_groups'
    params_files_list = [path.join(dir, t.resName) + '.params' for _, t in
                         process_networks.fxnl_groups.items()]

    opts = ['-ignore_waters false', '-mute core',
            '-extra_res_fa {}'.format(' '.join(params_files_list))]
    pyrosetta.init(extra_options=' '.join(opts),
                   set_logging_handler=None)

    # for now I'll just focus on a single file
    hash_tables = {}
    hash_info = {}
    fname = '{}_0.1_2.0_16.0.pkl'
    for t in process_networks.interaction_types:
        fn = fname.format(t)
        table_data = _fname_to_HTD(fn)
        hash_info[t] = table_data
        with open(fn, 'rb') as f:
            hash_tables[t] = pickle.load(f)

    # ok, let's grab something silly to make a good test:
    table = 'donor_donor'
    hashed_rays = next(iter(hash_tables[table].keys()))
    positioning_info = hash_tables[table][hashed_rays]

    fxnl_grps = list(process_networks.fxnl_groups.keys())
    xform = get_ht_from_table(positioning_info[1], hash_info[table].cart_resl,
                              hash_info[table].ori_resl,
                              hash_info[table].cart_bound)

    rsd = fxnl_grps[positioning_info[0]]
    pos_grp = process_networks.fxnl_groups[rsd]
    p = pyrosetta.Pose()
    pyrosetta.make_pose_from_sequence(p, 'Z[{}]'.format(rsd), 'fa_standard')
    coords = [np.array([*p.residue(1).xyz(atom)]) for atom in pos_grp.atoms]
    c = np.stack(coords)
    pos_frame = np.linalg.inv(hbond_ray_pairs.get_frame_for_coords(c))

    xf = np.dot(pos_frame, xform)
    tranform_pose(p, xf)
    # convert pose to ATOM records append all together and write them to a file


if __name__ == '__main__':
    main(sys.argv)
