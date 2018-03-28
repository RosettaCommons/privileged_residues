#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import pytest

from glob import glob
from os import path

pdb = path.expanduser("~weitzner/three_res_hbnet/arrangements/imidazole_carboxylate_amide/functional_stubs.pdb")

def _network_bidentate_placement(inpdb):
    import pyrosetta
    import privileged_residues
    import numpy as np

    from numpy.testing import assert_allclose
    from privileged_residues import bidentify as bd
    from privileged_residues.privileged_residues import _init_pyrosetta as init

    init()

    params = (0.1, 2.0, 16.0)

    atoms = privileged_residues.process_networks.fxnl_groups['A__'].atoms
    p = next(privileged_residues.process_networks.poses_for_all_models(inpdb))

    sp = p.split_by_chain(1)
    print(sp)

    hits = []
    ht_name = path.expanduser("~weitzner/three_res_hbnet/hash_tables/imidazole_carboxylate_amide_donor_donor_0.1_2.0_16.0.pkl")
    with open(ht_name, "rb") as f:
        table = pickle.load(f)
        rays = bd.look_up_connected_network(sp)
        hits += bd.look_up_interactions(rays, table, *params)

    ref_coords = np.stack([np.array([*p.residues[2].xyz(atom)]) for atom in atoms])

    for i, hit in enumerate(hits):
        coords = np.stack([np.array([*hit.residues[1].xyz(atom)]) for atom in atoms])
        try:
            assert_allclose(coords, ref_coords, atol=0.5)
            return
        except AssertionError:
            continue
    assert(False)


def test_network_bidentate_placement():
    _network_bidentate_placement(pdb)
