import privileged_residues
import pyrosetta

import numpy as np
import os
import random
import unittest

from os import path
from privileged_residues import chemical
from rmsd import rmsd

class Bidentates(unittest.TestCase):
    def bidentate(self, suffix):
        pdb_dir = "/home/weitzner/bidentate_hbond_pdbs_2/00_hashHBs_Sc_%s/pdb/" % (suffix)
        prefix = random.choice(os.listdir(pdb_dir))
        test_file = random.choice(os.listdir(path.join(pdb_dir, prefix)))
        test_file = path.join(pdb_dir, prefix, test_file)
        pose = next(privileged_residues.util.models_from_pdb(test_file))
        print(test_file)

        selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
        pres = privileged_residues.PrivilegedResidues()

        selector.append_index(5)

        ref_res = pose.residue(2)
        ref_coords = np.stack([ref_res.atom(n).xyz() for n in chemical.rsd_to_fxnl_grp[ref_res.name()].atoms])

        for p in pres.search(pose, ["sc_%s" % (suffix.lower())], selector):
            res = p.residue(1)
            coords = np.stack([res.atom(n).xyz() for n in chemical.functional_groups[res.name()].atoms])

            RMSD = rmsd(ref_coords, coords)
            if (RMSD < 0.25):
                return

        assert(False)

    def test_sc_sc(self):
        self.bidentate("Sc")

    def test_sc_scbb(self):
        self.bidentate("ScBb")

    def test_sc_bb(self):
        self.bidentate("Bb")

