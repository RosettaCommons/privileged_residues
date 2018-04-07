#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import pytest

from glob import glob
from os import path
from privileged_residues.privileged_residues import HAVE_PYROSETTA


pdb = path.expanduser("~weitzner/three_res_hbnet/arrangements/imidazole_carboxylate_amide/functional_stubs.pdb")

# workaround because Rosetta SFX does not capture all hbonds
def _look_up_connected_network(p):
    import itertools
    import privileged_residues

    from privileged_residues import hbond_ray_pairs

    pairs_of_rays = []

    resA = p.residue(2)
    resB = p.residue(1)

    listA = list(resA.Hpos_polar_sc()) + list(resA.accpt_pos_sc())
    listB = list(resB.Hpos_polar_sc()) + list(resB.accpt_pos_sc())

    for (j, k) in itertools.product(listA, listB):
        first = hbond_ray_pairs.create_ray(resA.xyz(j), resA.xyz(resA.atom_base(j)))
        second = hbond_ray_pairs.create_ray(resB.xyz(k), resB.xyz(resB.atom_base(k)))

        pairs_of_rays += [(first, second)]
        pairs_of_rays += [(second, first)]

    return pairs_of_rays

def _network_bidentate_placement(inpdb):
    import pyrosetta
    import privileged_residues
    import numpy as np
    import itertools

    from numpy.testing import assert_allclose
    from privileged_residues import bidentify as bd
    from privileged_residues import process_networks
    from privileged_residues.privileged_residues import _init_pyrosetta as init
    from pyrosetta.rosetta.core.kinematics import Stub

    init()

    params = (0.1, 2.0, 16.0)

    for pdb in glob(path.expanduser("~weitzner/three_res_hbnet/arrangements/*/functional_stubs.pdb")):
        print(pdb)

        p = next(privileged_residues.process_networks.poses_for_all_models(pdb))
        target = p.residue(3)
        atoms = privileged_residues.process_networks.fxnl_groups[target.name()].atoms
        
        hbond_types = ["acceptor", "donor"]

        interaction = path.basename(path.dirname((pdb)))
        ht_name_base = path.expanduser("~onalant/source/privileged_residues/local/hash_tables/%s/%s*_%s_%s_0.1_2.0_16.0.pkl")
        
        rays = _look_up_connected_network(p)
        
        found = False
        for (i, j) in itertools.product(hbond_types, hbond_types):
            hits = []
            ht_glob = glob(ht_name_base % (interaction, interaction, i, j))
            
            if (len(ht_glob) == 0):
                print("Could not find table")
                continue
                
            ht_name = ht_glob[0]

            with open(ht_name, "rb") as f:
                table = pickle.load(f)
                try:
                    hits += bd.look_up_interactions(rays, table, *params)
                except:
                    print("RIF XformHash assertion")
                    continue

            # NOTE(onalant): this needs to be pyrosetta distance compatible
            
            ref_coords = Stub(*(target.xyz(atom) for atom in privileged_residues.process_networks.fxnl_groups[target.name()].atoms))

            for k, hit in enumerate(hits):
                coords = Stub(*(hit.residue(1).xyz(atom) for atom in process_networks.fxnl_groups[hit.residue(1).name()].atoms))
                dist = pyrosetta.rosetta.core.kinematics.distance(ref_coords, coords)
                found = found or dist < 1.0
            
        assert(found)
@pytest.mark.skipif('not HAVE_PYROSETTA')
def test_network_bidentate_placement():
    _network_bidentate_placement(pdb)

