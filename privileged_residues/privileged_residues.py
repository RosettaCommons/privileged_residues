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

_PYROSETTA_INIT = False


def _init_pyrosetta():
    # initialize pyrosetta
    assert(HAVE_PYROSETTA)
    global _PYROSETTA_INIT
    if _PYROSETTA_INIT:
        return

    from . import process_networks as pn
    params_files_list = [path.join(_dir, t.resName) + '.params' for _, t in
                         pn.fxnl_groups.items()]
    opts = ['-ignore_waters false', '-mute core',
            '-extra_res_fa {}'.format(' '.join(params_files_list))]
    pyrosetta.init(extra_options=' '.join(opts),
                   set_logging_handler=_logging_handler)
    _PYROSETTA_INIT = True


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



def hash_all_network_files_in_directory(root_dir, out_dir='hash_tables',
                                        fname='functional_stubs.pdb',
                                        cart_resl=.1, ori_resl=2.,
                                        cart_bound=16.):
    """

    Args:
        fname (str):
        out_dir (str):
        fname (str): A pattern for network files in the subdirectories
            of root_dir.
        cart_resl (float): The cartesian resolution of the BCC lattice.
            Defaults to 0.1.
        ori_resl (float): The orientational resolution of the BCC
            lattice. In general, ori_resl should be roughly
            20 * cart_resl. Defaults to 2.0.
        cart_bound (float): The bounds of the lattice. Defaults to 16.

    """
    from os import path, makedirs, walk

    _init_pyrosetta()

    # get a list of the files that need to be processed
    arrangement_files = []
    output_file_base_names = []
    for directory, sub_dirs, files in walk(path.abspath(root_dir)):
        if fname in files:
            _, arr = path.split(directory)
            arrangement_files.append(path.join(directory, fname))
            output_file_base_names.append(path.join(out_dir),
                path.basename(path.normpath(directory)))

    if not path.exists(out_dir):
        makedirs(out_dir)

    # iterate over all of the files and make a giant ndarry containing all
    # of the information. This thing is going to take up some serious memory.
    for ntwrk_file in arrangement_files:
        hash_networks_and_write_to_file(fname, out_dir, cart_resl, ori_resl,
                                        cart_bound)

def _hash_from_file(fname, out_name_base, cart_resl, ori_resl, cart_bound, mod):
    import numpy as np
    import pickle
    from . import process_networks as pn

    _init_pyrosetta()

    hash_types = []
    print(fname)
    for i, pose in enumerate(pn.poses_for_all_models(fname)):
        if not i % 100:
            print('Pose ' + str(i))
        hash_types.extend(mod.find_all_relevant_hbonds_for_pose(pose))

    if hash_types == []:
        return
    # hash all of the processed infromation
    ht = pn.hash_full(np.stack(hash_types), cart_resl, ori_resl, cart_bound)

    # split the table into smaller tables based on interaction types
    # and write to disk as a pickled dictionary. Keys are the hashed ray pairs,
    # values are a list of (id, bin)s
    for i, interaction_type in enumerate(mod.interaction_types):
        t = ht[np.isin(ht['it'], [i])]
        out_fname = out_name_base + '_' + '_'.join([interaction_type,
                                                    str(cart_resl),
                                                    str(ori_resl),
                                                    str(cart_bound)]) + '.pkl'

        with open(out_fname, 'wb') as f:
            fdata = {}
            for k, v in zip(t['key'], t[['id', 'hashed_ht']]):
                try:
                    fdata[k].add(tuple(v))
                except KeyError:
                    fdata[k] = {tuple(v)}
            pickle.dump(fdata, f)


def hash_networks_and_write_to_file(fname, out_name_base, cart_resl=.1,
                                    ori_resl=2., cart_bound=16.):
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
    from . import process_networks as pn
    _hash_from_file(fname, out_name_base, cart_resl, ori_resl, cart_bound, pn)


def hash_bidentates_and_write_to_file(fname, out_name_base, cart_resl=.1,
                                      ori_resl=2., cart_bound=16.):
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
    from . import process_bidentates as pb
    _hash_from_file(fname, out_name_base, cart_resl, ori_resl, cart_bound, pb) 


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

    _init_pyrosetta()

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

    # now positioning_info is a set of possible values
    fxnl_grps = list(process_networks.fxnl_groups.keys())
    xform = pr.get_ht_from_table(positioning_info[0][1],
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
    pr.transform_pose(p, xf)
    # convert pose to ATOM records append all together and write them to a file


def find_privileged_interactions_in_pose(p):
    """
    """
    from os import path
    import pickle
    from . import bidentify as bd
    from . import position_residue as pr

    ht_name = 'Sc_Bb_02_0002_0000_Sc_Bb_0.1_2.0_16.0.pkl'
    ht_path = path.expanduser('~weitzner/bidentate_hbond_pdbs_2/hash_tables/Sc_Bb/02')
    ht_name_full = path.join(ht_path, ht_name)
    table_data = pr._fname_to_HTD('Sc_Bb_0.1_2.0_16.0.pkl')
    
    with open(ht_name_full, 'rb') as f:
        ht = pickle.load(f)
    pairs_of_rays = bd.find_hashable_positions(p, ht)

    return pairs_of_rays, ht

