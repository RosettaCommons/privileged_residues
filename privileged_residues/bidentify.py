import os
import numpy as np
import itertools

# The following packages are not pip-installable
# The import calls are wrapped in a try/except block
try:
    import pyrosetta
    from pyrosetta.rosetta.core.id import AtomID
except ImportError:
    print('Module "pyrosetta" not found in the current environment! '
          'Go to http://www.pyrosetta.org to download it.')
    pass


from . import hbond_ray_pairs
from . import position_residue as pr
from . import process_networks as pn

# the object of this submodule is to scan over all surface residues and 
# identify pairs of rays to look up in each hash table
def look_up_interactions(pairs_of_rays, ht, cart_resl, ori_resl, cart_bound):
    """
    """
    hits = []
    for r1, r2 in pairs_of_rays:
        hashed_rays = int(hbond_ray_pairs.hash_rays(r1, r2))
        try:
            positioning_info = ht[hashed_rays]
            print(positioning_info)
        except KeyError:
            continue
            
        ray_frame = hbond_ray_pairs.get_frame_for_rays(r1, r2)
        # now positioning_info is a set of possible values
        fxnl_grps = list(pn.fxnl_groups.keys())

        for pos_info in positioning_info:
            xform = None
            try:
                xform = pr.get_ht_from_table(pos_info[1],
                                             cart_resl,
                                             ori_resl,
                                             cart_bound)
            except:
                continue
                
            rsd = fxnl_grps[pos_info[0]]
            pos_grp = pn.fxnl_groups[rsd]

            p = pyrosetta.Pose()
            pyrosetta.make_pose_from_sequence(p, 'Z[{}]'.format(rsd), 'fa_standard')
            coords = [np.array([*p.residues[1].xyz(atom)]) for atom in pos_grp.atoms]
            c = np.stack(coords)

            pos_frame = hbond_ray_pairs.get_frame_for_coords(c)
            xf = np.dot(np.dot(ray_frame, xform), np.linalg.inv(pos_frame))
            pr.transform_pose(p, xf)
            hits.append(p)
    return hits    


def look_for_sc_bb_bidentates(p):
    """
    """
    
    pairs_of_rays = []
    for i in range(1, len(p.residues) + 1):
        # look at the amino and carboxyl rays, swap the order and look ahead
        # and back to see if the previous residue's carboxyl or the next 
        # residue's amino ray can be used to place a bidentate interaction


        rsd = p.residues[i]
        
        # ray reference atoms in backbone
        H = rsd.attached_H_begin(rsd.atom_index("N"))
        N = rsd.atom_base(H)
        O = rsd.atom_index("O")
        C = rsd.atom_base(O)

        n_ray = hbond_ray_pairs.create_ray(rsd.xyz(H), rsd.xyz(N))
        c_ray = hbond_ray_pairs.create_ray(rsd.xyz(O), rsd.xyz(C))

        pairs_of_rays += [(n_ray, c_ray), (c_ray, n_ray)]

        if i != 1:
            prev_rsd = p.residues[i - 1]

            O = prev_rsd.atom_index("O")
            C = prev_rsd.first_adjacent_heavy_atom(O)

            prev_c = hbond_ray_pairs.create_ray(prev_rsd.xyz(O), 
                                                prev_rsd.xyz(C))
            pairs_of_rays += [(n_ray, prev_c), (prev_c, n_ray)]

        if i != len(p.residues):
            next_rsd = p.residues[i + 1]
            H = next_rsd.attached_H_begin(rsd.atom_index("N"))
            N = next_rsd.first_adjacent_heavy_atom(H)
            next_n = hbond_ray_pairs.create_ray(next_rsd.xyz(H), 
                                                next_rsd.xyz(N))
            pairs_of_rays += [(next_n, c_ray), (c_ray, next_n)]
    return pairs_of_rays

def look_for_sc_scbb_bidentates(p):
    """
    """
    
    pairs_of_rays = []
    for rsd in p.residues:
        H = rsd.attached_H_begin(rsd.atom_index("N"))
        N = rsd.atom_base(H)
        O = rsd.atom_index("O")
        C = rsd.atom_base(O)

        n_ray = hbond_ray_pairs.create_ray(rsd.xyz(H), rsd.xyz(N))
        c_ray = hbond_ray_pairs.create_ray(rsd.xyz(O), rsd.xyz(C))

        for i in rsd.Hpos_polar_sc():
            ray = hbond_ray_pairs.create_ray(rsd.xyz(i), rsd.xyz(rsd.atom_base(i)))

            pairs_of_rays += [(n_ray, ray), (ray, n_ray)]
            pairs_of_rays += [(c_ray, ray), (ray, c_ray)]

        for i in rsd.accpt_pos_sc():
            ray = hbond_ray_pairs.create_ray(rsd.xyz(i), rsd.xyz(rsd.atom_base(i)))
            
            pairs_of_rays += [(n_ray, ray), (ray, n_ray)]
            pairs_of_rays += [(c_ray, ray), (ray, c_ray)]

    return pairs_of_rays

def look_for_sc_sc_bidentates(p):
    """
    """

    pairs_of_rays = []
    for rsd in p.residues:
        for (i, j) in itertools.permutations(list(rsd.Hpos_polar_sc()) + list(rsd.accpt_pos_sc()), 2):
            if i == j:
                continue

            first = hbond_ray_pairs.create_ray(rsd.xyz(i), rsd.xyz(rsd.atom_base(i)))
            second = hbond_ray_pairs.create_ray(rsd.xyz(j), rsd.xyz(rsd.atom_base(j)))

            pairs_of_rays += [(first, second)]

    return pairs_of_rays

def look_up_connected_network(p):
    """
    """

    pairs_of_rays = []

    hbondset = hbond_ray_pairs.identify_bonded_pairs(p, hbond_ray_pairs.find_hbonds(p))
    
    for hbond in hbondset.hbonds():
        don_res = p.residue(hbond.don_res())
        acc_res = p.residue(hbond.acc_res())
                
        don_list = list(don_res.Hpos_polar_sc()) + list(don_res.accpt_pos_sc())
        acc_list = list(acc_res.Hpos_polar_sc()) + list(acc_res.accpt_pos_sc())
        
        for (j, k) in itertools.product(don_list, acc_list):
            if (j == hbond.don_hatm() or k == hbond.acc_atm()):
                continue
            
            print(*((y, x.atom_name(x.atom_base(y))) for (x, y) in zip([don_res, acc_res], [j, k])))

            first = hbond_ray_pairs.create_ray(don_res.xyz(j), don_res.xyz(don_res.atom_base(j)))
            second = hbond_ray_pairs.create_ray(acc_res.xyz(k), acc_res.xyz(acc_res.atom_base(k)))
            
            pairs_of_rays += [(first, second)]
            pairs_of_rays += [(second, first)]

    return pairs_of_rays

def find_hashable_positions(p, ht):
    # look up each type of privileged interaction
    #
    hits = []
    pairs_of_rays = []

    # networks:
    #     iterate over pairs of residues with side chains that are 
    #     already hydrogen bonded to one another
    # Sc_Sc bidentates:
    #     iteratate over every residue, throw out non-bidentate capable 
    #     residues
    pairs_of_rays += look_for_sc_sc_bidentates(p)
    # Sc_ScBb bidentates:
    #     iterate over pairs of neighboring residues on the surface
    pairs_of_rays += look_for_sc_scbb_bidentates(p)
    # Sc_Bb bidentates:
    #     iterate over pairs of neighboring residues on the surface
    pairs_of_rays += look_for_sc_bb_bidentates(p)
    # hits += look_up_interactions(pairs_of_rays, ht)
    pairs_of_rays += look_up_connected_network(p)
    
    # tmp
    return pairs_of_rays

