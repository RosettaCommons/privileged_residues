# -*- coding: utf-8 -*-
"""Main module."""
import logging

# The following packages are not pip-installable
# The import calls are wrapped in a try/except block
try:
    import pyrosetta
    HAVE_PYROSETTA = True
except ImportError:
    print('Module "pyrosetta" not found in the current environment! '
          'Go to http://www.pyrosetta.org to download it.')
    HAVE_PYROSETTA = False
    pass

from os import path
_dir = path.join(path.dirname(__file__), 'data', 'functional_groups')

logging.basicConfig(level=logging.WARN)
_logging_handler = 'interactive'


def hash_ray_pairs_from_pdb_file(argv):
    """
    """
    from . import hbond_ray_pairs as hrp

    opts = ['-add_orbitals']
    target_fname = '../DHR10_133144_1.pdb'  # test!

    pyrosetta.init(extra_options=' '.join(opts),
                   set_logging_handler=_logging_handler)
    p = pyrosetta.pose_from_file(target_fname)
    hbset = hrp.find_hbonds(p)
    surf_hbset = hrp.identify_bonded_pairs(p, hbset)

    # make a list of the rays and then concatenate them with np.stack()
    donors = []
    acceptors = []
    for hbond in surf_hbset.hbonds():
        don_ray, acc_ray = hrp.find_ray_pair_for_hbond(p, hbond)
        donors.append(don_ray)
        acceptors.append(acc_ray)

    hrp.hash_rays(np.stack(donors), np.stack(acceptors))


def hash_networks_and_write_to_file(root_dir, out_dir,
                                    fname='functional_stubs.pdb',
                                    cart_resl=.1, ori_resl=2., cart_bound=16.):
    """

    Args:
        fname (str):
        out_dir (str):
        cart_resl (float): The cartesian resolution of the BCC lattice.
            Defaults to 0.1.
        ori_resl (float): The orientational resolution of the BCC
            lattice. In general, ori_resl should be roughly
            20 * cart_resl. Defaults to 2.0.
        cart_bound (float): The bounds of the lattice. Defaults to 16.
    """
    import numpy as np
    import pickle
    from os import path, makedirs, walk
    from . import process_networks as pn
    # TODO: use argparse to set these variables
    # TODO: adapt this script so all arrangements are searched in a single
    # execution. This will ensure that the groupings are appropriate
    # fname = root_dir + '/imidazole_carboxylate_guanidinium/' + fname
    out_dir = 'hash_tables'

    arrangement_files = []
    for directory, sub_dirs, files in walk(path.abspath(root_dir)):
        if fname in files:
            _, arr = path.split(directory)
            arrangement_files.append(path.join(directory, fname))

    print(arrangement_files)
    import sys
    sys.exit()

    assert(path.isfile(fname))
    if not path.exists(out_dir):
        makedirs(out_dir)

    params_files_list = [path.join(_dir, t.resName) + '.params' for _, t in
                         pn.fxnl_groups.items()]

    opts = ['-ignore_waters false', '-mute core',
            '-extra_res_fa {}'.format(' '.join(params_files_list))]
    pyrosetta.init(extra_options=' '.join(opts),
                   set_logging_handler=_logging_handler)

    hash_types = []
    for pose in pn.poses_for_all_models(fname):
        hash_types.extend(pn.find_all_relevant_hbonds_for_pose(pose))

    ht = pn.hash_full(np.stack(hash_types), cart_resl, ori_resl, cart_bound)

    for i, interaction_type in enumerate(pn.interaction_types):
        t = ht[np.isin(ht['it'], [i])]
        out_fname = '_'.join([interaction_type, str(cart_resl), str(ori_resl),
                              str(cart_bound)]) + '.pkl'

        with open(path.join(out_dir, out_fname), 'wb') as f:
            pickle.dump({k: v for k, v in zip(t['key'],
                                              t[['id', 'hashed_ht']])}, f)


def laod_hash_tables_from_disk(fname=None):
    """Load the hash tables (pickled dictionaries) containing hydrogen
    bond ray pair hashes as keys and a functional group identifier and
    a bin number of Body Centered  Cubic (BCC) lattice corresponding to
    a homogenous transform that will place the functional group relative
    to the hydrogen bonding rays as values. In its current state, this
    script selects an arbitrary item from the table and ensures that a
    functional group can be placed correctly.

    Args:
        fname (str):

    """
    import numpy as np
    import pickle
    from os import path
    from . import process_networks
    from . import position_residue as pr

    # setup per-use variables with argparse
    params_files_list = [path.join(_dir, t.resName) + '.params' for _, t in
                         process_networks.fxnl_groups.items()]

    opts = ['-ignore_waters false', '-mute core',
            '-extra_res_fa {}'.format(' '.join(params_files_list))]
    pyrosetta.init(extra_options=' '.join(opts),
                   set_logging_handler=_logging_handler)

    # for now I'll just focus on a single file
    hash_tables = {}
    hash_info = {}
    fname = '{}_0.1_2.0_16.0.pkl'
    for t in process_networks.interaction_types:
        fn = fname.format(t)
        table_data = pr._fname_to_HTD(fn)
        hash_info[t] = table_data
        with open(fn, 'rb') as f:
            hash_tables[t] = pickle.load(f)

    # ok, let's grab something silly to make a good test:
    table = 'donor_donor'
    hashed_rays = next(iter(hash_tables[table].keys()))
    positioning_info = hash_tables[table][hashed_rays]

    fxnl_grps = list(process_networks.fxnl_groups.keys())
    xform = pr.get_ht_from_table(positioning_info[1],
                                 hash_info[table].cart_resl,
                                 hash_info[table].ori_resl,
                                 hash_info[table].cart_bound)

    rsd = fxnl_grps[positioning_info[0]]
    pos_grp = process_networks.fxnl_groups[rsd]
    p = pyrosetta.Pose()
    pyrosetta.make_pose_from_sequence(p, 'Z[{}]'.format(rsd), 'fa_standard')
    coords = [np.array([*p.residues[1].xyz(atom)]) for atom in pos_grp.atoms]
    c = np.stack(coords)
    pos_frame = np.linalg.inv(hbond_ray_pairs.get_frame_for_coords(c))

    xf = np.dot(pos_frame, xform)
    tranform_pose(p, xf)
    # convert pose to ATOM records append all together and write them to a file
