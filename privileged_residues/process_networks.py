import argparse
import numpy as np
import sys
import pickle

from collections import namedtuple, OrderedDict
from itertools import permutations
from more_itertools import chunked
from os import makedirs, path

# The following packages are not pip-installable
# The import calls are wrapped in a try/except block
try:
    import pyrosetta
except ImportError:
    print('Module "pyrosetta" not found in the current environment! '
          'Go to http://www.pyrosetta.org to download it.')
    pass

try:
    import rif
    from rif.hash import *
except ImportError:
    print('Module "rif" not found in the current environment! '
          'Go to https://github.com/willsheffler/rif for more information.')
    pass

from . import hbond_ray_pairs

# the order of the keys of this dictionary
FxnlGrp = namedtuple('FxnlGrp', ['resName', 'donor', 'acceptor', 'atoms'])
fxnl_groups = OrderedDict(sorted({'OH_': FxnlGrp('hydroxide', True, True,
                                                 ['CV', 'OH', 'HH']),
                                  'G__': FxnlGrp('guanidium', True, False,
                                                 ['CZ', 'NH1', 'NH2']),
                                  'I__': FxnlGrp('imidazole', True, True,
                                                 ['ND1', 'CD2', 'NE2']),
                                  # imidazole tautomer
                                  'ID_': FxnlGrp('imidazole_D', True, True,
                                                 ['ND1', 'CD2', 'NE2']),
                                  'A__': FxnlGrp('amine', True, False,
                                                 ['NZ', '1HZ', '2HZ']),
                                  'C__': FxnlGrp('carboxylate', False, True,
                                                 ['CD', 'OE1', 'OE2']),
                                  'CA_': FxnlGrp('carboxamide', True, True,
                                                 ['CG', 'OD1', 'ND2']),
                                  }.items(), key=lambda t: t[0]))

interaction_types = ['acceptor_acceptor', 'acceptor_donor', 'donor_acceptor',
                     'donor_donor']

Interaction = namedtuple('Interaction', ['pos', 'partner'])
AtomIDPair = namedtuple('AtomIDPair', ['center', 'base'])


def get_models_from_file(fname):
    """Read a PDB-formatted file and return a list of ATOM records
    grouped by model.

    Notes:
        ATOM records are represented by a fixed-width format described
        here_.

        .. _here: http://www.wwpdb.org/documentation/\
        file-format-content/format33/sect9.html#ATOM

    Args:
        fname (str): The name of a PDB-formatted file.

    Returns:
        list: A list of lists of ATOM records that represent the
        models.
    """
    with open(fname, 'r') as f:
        atom_records = [l.rstrip() for l in f.readlines()]

    models = []
    current_model = []
    for record in atom_records:
        current_model.append(record)
        if record == 'ENDMDL':
            models.append(current_model)
            current_model = []

    # if the last line in the file is not 'ENDMDL' it needs to be added to the
    # list here.
    if len(current_model) != 0:
        models.append(current_model)
    return models


def pose_from_atom_records(atom_recs):
    """Create a Pose from a list of ATOM records and return it.

    Notes:
        ATOM records are represented by a fixed-width format described
        here_.

        .. _here: http://www.wwpdb.org/documentation/\
        file-format-content/format33/sect9.html#ATOM

    Args:
        atom_recs (list): A list of ATOM records (str) describing a
            particular conformation of a protein or other macromolecule
            that can be represented as a Pose.

    Returns:
        pyrosetta.Pose: The Pose containing the information in the ATOM
        records.
    """
    p = pyrosetta.Pose()

    from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
    pose_from_pdbstring(p, pdbcontents='\n'.join(atom_recs))
    return p


def poses_for_all_models(fname):
    """

    Args:
        fname (str):

    Yields:
        pyrosetta.Pose:
    """
    for model in get_models_from_file(fname):
        yield pose_from_atom_records(model)


def generate_combinations(n):
    """

    Args:
        n (int):

    Yields:
        tuple:
    """
    # this ensures everything is in a usable order
    for i, j in chunked(permutations(range(1, n + 1), 2), 2):
        yield Interaction(*i), Interaction(*j)


def rays_for_interaction(donor, acceptor, interaction):
    """

    Args:
        donor (pyrosetta.rosetta.core.conformation.Residue):
        acceptor (pyrosetta.rosetta.core.conformation.Residue):
        interaction (Interaction):

    Returns:
        tuple: A tuple of rays represented as (2, 4) numpy.arrays
        describing the stationary and positioned residues,
        respectively.
    """
    from pyrosetta.rosetta.core.id import AtomID

    def _get_atom_id_pair(rsd, atmno):
        base_atmno = rsd.atom_base(atmno)
        return AtomIDPair(AtomID(atmno, rsd.seqpos()),
                          AtomID(base_atmno, rsd.seqpos()))

    def _vector_from_id(atom_id, rsd):
        return np.array([*rsd.xyz(atom_id.atomno())])

    def _positioning_ray(rsd, atmno):
        id_pair = _get_atom_id_pair(rsd, atmno)
        return hbond_ray_pairs.create_ray(_vector_from_id(id_pair.center, rsd),
                                          _vector_from_id(id_pair.base, rsd))

    don_rays = []
    for i in range(1, donor.natoms() + 1):
        # in case we use full residues later on
        if donor.atom_is_backbone(i):
            continue
        if donor.atom_is_polar_hydrogen(i):
            don_rays.append(_positioning_ray(donor, i))

    acc_rays = []
    for i in acceptor.accpt_pos():
        # in case we use full residues later on
        if acceptor.atom_is_backbone(i):
            continue
        acc_rays.append(_positioning_ray(acceptor, i))

    don = np.stack(don_rays)
    acc = np.stack(acc_rays)
    dist = np.linalg.norm(don[:, np.newaxis, 0, :] - acc[np.newaxis, :, 0, :],
                          axis=-1)
    don_idx, acc_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    if interaction.pos == donor.seqpos():
        target = acc_rays[acc_idx]
        positioned_residue = don_rays[don_idx]
    else:
        target = don_rays[don_idx]
        positioned_residue = acc_rays[acc_idx]

    return target, positioned_residue


def find_all_relevant_hbonds_for_pose(p, hash_types):
    """

    Notes:
        Something about the dtype here.

    Args:
        p (pyrosetta.Pose):
        hash_types (list):

    Returns:
        list:
    """
    # ah, at last. this is where it gets a little tricky.
    # everything should be connected to everything else...
    #  , ray 1, ray 2, positioned rsd id, homogeneous transform to position rsd
    entry_type = np.dtype([('it', 'u8'), ('r1', 'f8', (2, 4)),
                           ('r2', 'f8', (2, 4)), ('id', 'u8'),
                           ('ht', 'f8', (4, 4))])

    res_orderings = generate_combinations(len(p.residues))

    first = []
    second = []

    fxnl_grps = list(fxnl_groups.keys())

    for interactions in res_orderings:
        # look at the order, figure out which hydrogen bonds to use
        rp = None
        ht = []
        for i, interaction in enumerate(interactions):
            pos_rsd = p.residue(interaction.pos)
            pos_fxnl_grp = fxnl_groups[pos_rsd.name3()]
            par_fxnl_grp = fxnl_groups[p.residue(interaction.partner).name3()]

            if pos_fxnl_grp is None:
                print(p.residue(interaction.pos))

            if par_fxnl_grp is None:
                print(p.residue(interaction.partner))

            # determine which residue is the donor and which is the acceptor
            res_nos = (interaction.pos, interaction.partner)
            if (par_fxnl_grp.donor and not par_fxnl_grp.acceptor) or \
               (pos_fxnl_grp.acceptor and not pos_fxnl_grp.donor):
                assert(pos_fxnl_grp.acceptor and par_fxnl_grp.donor)
                acceptor, donor = (p.residue(i) for i in res_nos)
                ht.append('donor')
            elif (par_fxnl_grp.acceptor and not par_fxnl_grp.donor) or \
                 (pos_fxnl_grp.donor and not pos_fxnl_grp.acceptor):
                assert(pos_fxnl_grp.donor and par_fxnl_grp.acceptor)
                donor, acceptor = (p.residue(i) for i in res_nos)
                ht.append('acceptor')
            else:
                print('Ambiguous arrangement: both can donate & accept!')
                print('Trying something a little more complicated...')
                print('Oh shit, I should probably implement this!')
                sys.exit(1)

            target, pos = rays_for_interaction(donor, acceptor, interaction)
            first.append(target) if i == 0 else second.append(target)

        table = '_'.join(ht)

        # now we just need the xform from the ray-frame to the positioned rsd
        ray_frame = hbond_ray_pairs.get_frame_for_rays(first[-1], second[-1])

        c = [np.array([*pos_rsd.xyz(atom)]) for atom in pos_fxnl_grp.atoms]
        pos_frame = hbond_ray_pairs.get_frame_for_coords(np.stack(c))
        frame_to_store = np.dot(np.linalg.inv(ray_frame), pos_frame)
        assert(np.allclose(pos_frame, np.dot(ray_frame, frame_to_store)))

        # store a tuple of the Rays and the positioning information.
        # pop first and second upon constructing entry to prepare for the
        # next iteration
        hash_types.append(np.array((interaction_types.index(table),
                                    first.pop(), second.pop(),
                                    fxnl_grps.index(pos_rsd.name3()),
                                    frame_to_store),
                                   dtype=entry_type))

    return hash_types


def hash_full(full_array, cart_resl=0.1, ori_resl=2., cart_bound=16.):
    """

    Notes:
        Something about the dtype here.

    Args:
        full_array (np.array):
        cart_resl (float):
        ori_resl (float):
        cart_bound (float):

    Returns:
        np.array:
    """
    # these are kind of magic numbers
    # together they define the grid universe in which we are living
    xh = XformHash_bt24_BCC6_X3f(cart_resl, ori_resl, cart_bound)
    # to go from key -> bin center (~the xform I care about) use...
    # xh.get_center([key])['raw'].squeeze()

    hashed_type = np.dtype([('it', 'u8'), ('key', 'u8'), ('id', 'u8'),
                            ('hashed_ht', 'u8')])

    result = np.empty_like(full_array, dtype=hashed_type)
    result['it'] = full_array['it']
    result['id'] = full_array['id']
    result['key'] = hbond_ray_pairs.hash_rays(full_array['r1'],
                                              full_array['r2'])

    stretched_xf = full_array['ht'].astype('f4').reshape(-1, 16).view(rif.X3)
    result['hashed_ht'] = xh.get_key(stretched_xf).squeeze()
    return result


def main(argv):
    """
    """
    # TODO: use argparse to set these variables
    # TODO: adapt this script so all arrangements are searched in a single
    # execution. This will ensure that the groupings are appropriate
    fn = 'arrangements/imidazole_carboxylate_guanidinium/functional_stubs.pdb'
    dir = 'functional_groups'
    out_dir = 'hash_tables'
    cart_resl, ori_resl, cart_bound = 0.1, 2., 16.

    assert(path.isfile(fn))
    if not path.exists(out_dir):
        makedirs(out_dir)

    params_files_list = [path.join(dir, t.resName) + '.params' for _, t in
                         fxnl_groups.items()]

    opts = ['-ignore_waters false', '-mute core',
            '-extra_res_fa {}'.format(' '.join(params_files_list))]
    pyrosetta.init(extra_options=' '.join(opts),
                   set_logging_handler=None)

    hash_types = []
    for pose in poses_for_all_models(fn):
        hash_types = find_all_relevant_hbonds_for_pose(pose, hash_types)

    ht = hash_full(np.stack(hash_types), cart_resl, ori_resl, cart_bound)

    for i, interaction_type in enumerate(interaction_types):
        t = ht[np.isin(ht['it'], [i])]
        out_fname = '_'.join([interaction_type, str(cart_resl), str(ori_resl),
                              str(cart_bound)]) + '.pkl'

        with open(path.join(out_dir, out_fname), 'wb') as f:
            pickle.dump({k: v for k, v in zip(t['key'],
                                              t[['id', 'hashed_ht']])}, f)


if __name__ == '__main__':
    main(sys.argv)
