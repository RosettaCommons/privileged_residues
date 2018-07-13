import privileged_residues
import pyrosetta

import numpy as np
import os
import random

from datetime import date
from os import path
from privileged_residues import chemical
from rmsd import rmsd

def test_sc_sc():
    pdb_dir = "/home/weitzner/bidentate_hbond_pdbs_2/00_hashHBs_Sc_Sc/pdb/"
    prefix = random.choice(os.listdir(pdb_dir))
    test_file = random.choice(os.listdir(path.join(pdb_dir, prefix)))
    test_file = path.join(pdb_dir, prefix, test_file)
    pose = next(privileged_residues.util.models_from_pdb(test_file))
    print(test_file)

    selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    pres = privileged_residues.PrivilegedResidues()

    selector.append_index(2)

    ref_res = pose.residue(5)
    ref_coords = np.stack([ref_res.atom(n).xyz() for n in chemical.rsd_to_fxnl_grp[ref_res.name()].atoms])

    for p in pres.search(pose, ["sc_sc"], selector):
        res = p.residue(1)
        coords = np.stack([res.atom(n).xyz() for n in chemical.functional_groups[res.name()].atoms])

        RMSD = rmsd(ref_coords, coords)
        if (RMSD < 0.5):
            return

    assert(False)

