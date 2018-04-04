import numpy as np

from collections import namedtuple, OrderedDict
from itertools import permutations
from more_itertools import chunked
from numpy.testing import assert_allclose

# The following packages are not pip-installable
# The import calls are wrapped in a try/except block
try:
    import pyrosetta
    from pyrosetta.rosetta.core.id import AtomID
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
from . import rotation_matrix


# the order of the keys of this dictionary
FxnlGrp = namedtuple('FxnlGrp', ['resName', 'donor', 'acceptor', 'atoms'])
FxnlGrp.__doc__ = """Store hydrogen bonding information about a
functional group as well as information that can be used to position it
in three-space.

Attributes:
    resName (str): Name of the functional group.
    donor (bool): True if the functional group can be a donor in a
        hydrogen bond.
    acceptor (bool): True if the functional group can be an acceptor in
        a hydrogen bond.
    atoms (list): A list of three atom names (str) that are used to
        construct a coordinate frame to describe the position of the
        functional group in space.
"""

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
Interaction.__doc__ = """Store a pair of interacting residues by
sequence position.

Attributes:
    pos (int): The sequence position of the positioned residue.
    partner (int): The sequence position of the stationary residue
        forming an interaction with the positioned residue.
"""

AtomIDPair = namedtuple('AtomIDPair', ['center', 'base'])
AtomIDPair.__doc__ = """Store a pair AtomIDs that can be used to
construct a ray in space to describe a particular geometry.

Notes:
    This type is used to construct rays that are centered at the
    `center` residue and have unit length in the `base`-`center`
    direction.

Attributes:
    center (pyrosetta.rosetta.core.id.AtomID): The atom on which to
        center a ray.
    base (pyrosetta.rosetta.core.id.AtomID): The atom used to define
        the direction of the ray.
"""


# helper functions for identifying hbond partners
def _get_atom_id_pair(rsd, atmno):
    base_atmno = rsd.atom_base(atmno)
    if rsd.name() == 'OH_' and atmno == 2:
        base_atmno = 1
    return AtomIDPair(AtomID(atmno, rsd.seqpos()),
                      AtomID(base_atmno, rsd.seqpos()))


def _vector_from_id(atom_id, rsd):
    return np.array([*rsd.xyz(atom_id.atomno())])


def _positioning_ray(rsd, atmno):
    id_pair = _get_atom_id_pair(rsd, atmno)
    return hbond_ray_pairs.create_ray(_vector_from_id(id_pair.center, rsd),
                                      _vector_from_id(id_pair.base, rsd))


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
        model = []
        for l in f:
            if (l.startswith("#")):
                continue
                
            line = l.rstrip()
            model.append(line)
            
            if (line.startswith("ENDMDL")):
                yield model
                model = []
        
        if (len(model) > 0):
            yield model


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
    """Generate a Pose for each model in a PDB-formatted file. Models
    are separated by an 'ENDMDL' entry in the file.

    Args:
        fname (str): The name of a PDB-formatted file.

    Yields:
        pyrosetta.Pose: The Pose representing the next model in the
        file.
    """
    for model in get_models_from_file(fname):
        yield pose_from_atom_records(model)


def generate_combinations(n):
    """Generate a tuple of Interactions the describes the connectivity
    of a closed network of residues.

    Args:
        n (int): The number of residues in the network.

    Yields:
        tuple: The Interaction instances that describe the
        connectivity of a network.
    """
    # this ensures everything is in a usable order
    for i in chunked(permutations(range(1, n + 1), 2), n - 1):
        yield (Interaction(*j) for j in i)


def rays_for_interaction(donor, acceptor, interaction):
    """Create a pair of rays describing a hydrogen and return them as a
    tuple.

    Args:
        donor (pyrosetta.rosetta.core.conformation.Residue): The
            residue containing the hydrogen bond donor.
        acceptor (pyrosetta.rosetta.core.conformation.Residue): The
            residue containing the hydrogen bond acceptor.
        interaction (Interaction): The interaction that describes which
            residue is stationary and which is psitioned by the
            interaction.

    Returns:
        tuple: A tuple of rays represented as (2, 4) numpy.arrays
        describing the stationary and positioned residues,
        respectively.
    """
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


def positioned_residue_is_donor(positioned, target):
    """Determine whether the residue positioned by the hydrogen bond of
    interest is the donor and return True if it is.

    Args:
        positioned (pyrosetta.rosetta.core.conformation.Residue): The
            residue that is positioned by the hydrogen bond.
        target (pyrosetta.rosetta.core.conformation.Residue): The
            residue that is having a hydrogen bonding ray extracted
            from it.
    Returns:
        bool: True if positioned residue is the hydrogen bond donor,
        False if it is the acceptor.

        If the function cannot determine which residue is the donor, it
        returns None.
    """
    def _get_all_rays(rsd):
        rays = []
        atms = []
        for i in range(1, rsd.natoms() + 1):
            # in case we use full residues later on
            if rsd.atom_is_backbone(i):
                continue
            if rsd.atom_type(i).element() in ('N', 'O', 'H'):
                rays.append(_positioning_ray(rsd, i))
                atms.append(i)
        return rays, atms

    tgt_rays, tgt_atms = _get_all_rays(target)
    pos_rays, pos_atms = _get_all_rays(positioned)

    tgt = np.stack(tgt_rays)
    pos = np.stack(pos_rays)
    dist = np.linalg.norm(tgt[:, np.newaxis, :, 0] - pos[np.newaxis, :, :, 0],
                          axis=-1)
    tgt_idx, pos_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    if positioned.atom_type(pos_atms[pos_idx]).element() == 'H':
        return True
    elif target.atom_type(tgt_atms[tgt_idx]).element() == 'H':
        return False
    return None


def find_all_relevant_hbonds_for_pose(p):
    """Enumerate all possible arrangements of residues in a Pose of a
    closed hydrogen-bonded network of residues, pack the arrangment
    into a numpy.array with enough information to reconstruct the
    arrangement and return the arrays as a list.

    Notes:
        The resulting numpy.arrays have a custom dtype with the
        following fields:

        1. it: The interation type.
        2. r1: Ray 1
        3. r2: Ray 2
        4. id: The identifier of the functional group positioned by
           this interaction.
        5. ht: The homogenous transform that describes the orientation
           of the positioned residue relative to the stationary residues.

    Args:
        p (pyrosetta.Pose): The Pose to examine.

    Returns:
        list: A list of numpy.arrays that each represent a network
        configuration.
    """
    # ah, at last. this is where it gets a little tricky.
    # everything should be connected to everything else...
    #  , ray 1, ray 2, positioned rsd id, homogeneous transform to position rsd
    entry_type = np.dtype([('it', 'u8'), ('r1', 'f8', (4, 2)),
                           ('r2', 'f8', (4, 2)), ('id', 'u8'),
                           ('ht', 'f8', (4, 4))])

    res_orderings = generate_combinations(len(p.residues))

    first = []
    second = []
    hash_types = []

    # it some poses have too many residues in them, presumably the result of 
    # some strange race condition in an upstream process. Since the data are
    # somewhat nonsensical, I am just going to skip these poses entirely.
    if len(p.residues) != 3:
        return hash_types

    fxnl_grps = list(fxnl_groups.keys())
    for interactions in res_orderings:
        # look at the order, figure out which hydrogen bonds to use
        rp = None
        ht = []
        for i, interaction in enumerate(interactions):
            pos_rsd = p.residues[interaction.pos]
            pos_fxnl_grp = fxnl_groups[pos_rsd.name3()]
            par_fxnl_grp = fxnl_groups[p.residues[interaction.partner].name3()]

            if pos_fxnl_grp is None:
                print(p.residues[interaction.pos])

            if par_fxnl_grp is None:
                print(p.residues[interaction.partner])

            # determine which residue is the donor and which is the acceptor
            res_nos = (interaction.pos, interaction.partner)
            if (par_fxnl_grp.donor and not par_fxnl_grp.acceptor) or \
               (pos_fxnl_grp.acceptor and not pos_fxnl_grp.donor):
                assert(pos_fxnl_grp.acceptor and par_fxnl_grp.donor)
                acceptor, donor = (p.residues[i] for i in res_nos)
                ht.append('donor')
            elif (par_fxnl_grp.acceptor and not par_fxnl_grp.donor) or \
                 (pos_fxnl_grp.donor and not pos_fxnl_grp.acceptor):
                assert(pos_fxnl_grp.donor and par_fxnl_grp.acceptor)
                donor, acceptor = (p.residues[i] for i in res_nos)
                ht.append('acceptor')
            else:
                par_rsd = p.residues[interaction.partner]
                pos_don = positioned_residue_is_donor(pos_rsd, par_rsd)
                assert(pos_don is not None)
                if pos_don:
                    donor, acceptor = pos_rsd, par_rsd
                    ht.append('acceptor')
                else:
                    acceptor, donor = pos_rsd, par_rsd
                    ht.append('donor')

            target, pos = rays_for_interaction(donor, acceptor, interaction)
            first.append(target) if i == 0 else second.append(target)

        table = '_'.join(ht)

        # now we just need the xform from the ray-frame to the positioned rsd
        ray_frame = hbond_ray_pairs.get_frame_for_rays(first[-1], second[-1])

        c = [np.array([*pos_rsd.xyz(atom)]) for atom in pos_fxnl_grp.atoms]
        pos_frame = hbond_ray_pairs.get_frame_for_coords(np.stack(c))
        frame_to_store = np.dot(np.linalg.inv(ray_frame), pos_frame)
        assert_allclose(pos_frame, np.dot(ray_frame, frame_to_store),
                        atol=1E-10)
        array_size = 1
        if pos_fxnl_grp.resName == 'hydroxide':
            # hydroxide only has two clearly positioned atoms
            # the positioned frame needs to be rotated about the OH--HH bond
            # to fill out the relevant orientations.
            # the rotation will be centered on the hydrogen.
            resl = 5.  # degrees

            rot_cntr = np.array([*pos_rsd.xyz('HH')])
            axis = np.array([*pos_rsd.xyz('OH')]) - rot_cntr
            angles = np.arange(0., 360., resl)
            r = rotation_matrix.rot_ein(axis, angles, degrees=True,
                                        center=rot_cntr)
            assert(r.shape == (int(360. / resl),) + (4, 4))

            frame_to_store = r * frame_to_store
            array_size = frame_to_store.shape[0]

        # store a tuple of the Rays and the positioning information.
        # pop first and second upon constructing entry to prepare for the
        # next iteration
        res = np.empty(array_size, dtype=entry_type)
        res['it'] = interaction_types.index(table)
        res['r1'] = first.pop()
        res['r2'] = second.pop()
        res['id'] = fxnl_grps.index(pos_rsd.name3())
        res['ht'] = frame_to_store
        hash_types.extend(res)

    return hash_types


def hash_full(full_array, cart_resl=0.1, ori_resl=2., cart_bound=16.):
    """Hash an input numpy.array with a structured dtype and return a
    new array.

    Notes:
        The input nump.array (full_array) is expected to have been
        generated by `find_all_relevant_hbonds_for_pose`. To prepare
        the array for this function, use `np.stack(list_of_arrays)`.

        The resulting numpy.array have a custom dtype with the
        following fields:

        1. it: The interation type.
        2. key: The hash of two rays in space.
        3. id: The identifier of the functional group positioned by
           this interaction.
        4. hashed_ht: The hash of a homogenous transform.

    Args:
        full_array (np.array): The full array of network
            configurations.
        cart_resl (float): The cartesian resolution of the BCC lattice
        ori_resl (float): The orientational resolution of the BCC
            lattice. In general, ori_resl should be roughly
            20 * cart_resl.
        cart_bound (float): The bounds of the lattice.

    Returns:
        np.array: The hased form of the input array.
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


if __name__ == '__main__':
    print('Don\'t execute me, bruh.')
