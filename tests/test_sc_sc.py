import privileged_residues
import pyrosetta

import os
import random

from os import path
from privileged_residues import postproc

# NOTE(onalant): /home/weitzner/bidentate_hbond_pdbs_2/00_hashHBs_Sc_Sc/pdb/00/debug_c1215_R_E_member_5ws7_A.ideal_bbbb0.00_scAbbB0.00_scBbbA0.00_scsc2.00_amsf1.25_bmsf1.17_5009_tripep_c8c9895da0902e7af70bab0e857c5a26032.pdb

pdb_dir = "/home/weitzner/bidentate_hbond_pdbs_2/00_hashHBs_Sc_Sc/pdb/"
prefix = random.choice(os.listdir(pdb_dir))
test_file = random.choice(os.listdir(path.join(pdb_dir, prefix)))
test_file = path.join(pdb_dir, prefix, test_file)
pose = pyrosetta.pose_from_pdb(test_file)

selector = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
pres = privileged_residues.PrivilegedResidues()

output = list(pres.search(pose, ["sc_sc"], selector))
for (n, pose) in enumerate(output):
	pose.dump_pdb("/home/onalant/dump/2018-06-29_PResiduesDump/foo%d.pdb" % (n))

print("DONE")

