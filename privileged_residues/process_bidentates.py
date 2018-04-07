import numpy as np

from collections import defaultdict, namedtuple, OrderedDict
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
from . import process_networks as pn

ResInfo = namedtuple('ResName', ['grp', 'atoms'])
rsd_to_fxnl_grp = {'ASN': ResInfo('CA_', ['CG', 'OD1', 'ND2']),
                   'GLN': ResInfo('CA_', ['CD', 'OE1', 'NE2']),
                   'ASP': ResInfo('C__', ['CG', 'OD1', 'OD2']),
                   'GLU': ResInfo('C__', ['CD', 'OE1', 'OE2']),
                   # Use HD1/HE2 to determine which tautomer to use
                   'HIS': ResInfo('I__', ['ND1', 'CD2', 'NE2']),
                   'ARG': ResInfo('G__', ['CZ', 'NH1', 'NH2']),
                   'SER': ResInfo('OH_', ['CB', 'OG', 'HG']),
                   'THR': ResInfo('OH_', ['CB', 'OG1', 'HG1']),
                   'TYR': ResInfo('OH_', ['CZ', 'OH', 'HH']),
                   'LYS': ResInfo('A__', ['NZ', '1HZ', '2HZ']),
                   }

interaction_types = ['Sc_Sc', 'Sc_ScBb', 'Sc_Bb']


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

    # we are only interested in the two best-scoring hyrogen bonds
    hbs = sorted(list(hbond_ray_pairs.find_hbonds(p, exclude_bsc=False, 
        exclude_scb=False).hbonds()), key=lambda hb: hb.energy())[:2]
    
    # figure out which residue is being postioned and which is stationary
    rsds = defaultdict(list)
    for hb in hbs:
        r = (hb.don_res(), hb.don_hatm_is_backbone()), \
            (hb.acc_res(), hb.acc_atm_is_backbone())
        for k, v in r:
            rsds[k].append(v) 

    res_nums = []
    n_bb = 0
    for k, v in rsds.items():
        # if the residue is being postioned, it should have exactly two hbonds
        # and neither should involve the backbone
        for e in v:
          n_bb += int(e)
        if len(v) != 2:
            continue
        if True not in v:
            res_nums.append(k)

    first = []
    second = []
    hash_types = []

    table = ''
    if n_bb == 0:
        table = 'Sc_Sc'
    elif n_bb == 1:
        table = 'Sc_ScBb'
    elif n_bb == 2:
        table = 'Sc_Bb'
    else:
        print('wtf, mate')
        sys.exit()

    fxnl_grps = list(pn.fxnl_groups.keys())

    for positioned in res_nums:
        # it's probably smart to do the reverse, too
        for i, hb in enumerate(hbs):
            '''
            par = hb.don_res() if positioned != hb.don_res() else hb.acc_res()
            interaction = pn.Interaction(positioned, par)
            target, pos = pn.rays_for_interaction(p.residues[hb.don_res()], 
                                                  p.residues[hb.acc_res()],
                                                  interaction)
            '''
            pos_rsd = p.residues[positioned]
            pos_fxnl_grp = rsd_to_fxnl_grp[pos_rsd.name3()]
            d, a = hbond_ray_pairs.find_ray_pair_for_hbond(p, hb)
            target, pos = (d, a) if positioned != hb.don_res() else (a, d)
            first.append(target) if i == 0 else second.append(target)


        ## NOTE: from here to the end of this function, the code is nearly identical to
        ## pn.find_all_relevant_hbonds_for_pose(p) -- factor that bit out?

        # now we just need the xform from the ray-frame to the positioned rsd
        ray_frame = hbond_ray_pairs.get_frame_for_rays(first[-1], second[-1])


        c = [np.array([*pos_rsd.xyz(atom)]) for atom in pos_fxnl_grp.atoms]
        pos_frame = hbond_ray_pairs.get_frame_for_coords(np.stack(c))
        frame_to_store = np.dot(np.linalg.inv(ray_frame), pos_frame)
        assert_allclose(pos_frame, np.dot(ray_frame, frame_to_store),
                        atol=1E-10)

        array_size = 1
        # store a tuple of the Rays and the positioning information.
        # pop first and second upon constructing entry to prepare for the
        # next iteration
        res = np.empty(array_size, dtype=entry_type)
        res['it'] = interaction_types.index(table)
        res['r1'] = first.pop()
        res['r2'] = second.pop()
        res['id'] = fxnl_grps.index(rsd_to_fxnl_grp[pos_rsd.name3()].grp)
        res['ht'] = frame_to_store
        hash_types.extend(res)

    return hash_types

    

