import privileged_residues
import pyrosetta

import os
import random

from datetime import date
from os import path

test_file = path.join(path.dirname(path.abspath(__file__)), "data", "sc_bb_example.pdb")
pose = pyrosetta.pose_from_pdb(test_file)

selector = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
pres = privileged_residues.PrivilegedResidues()

outdir = path.join("/home/onalant/dump/", "%s_PResidues" % (date.today().isoformat()))
os.makedirs(outdir, exist_ok=True)

for (n, p) in enumerate(pres.search(pose, ["sc_bb"], selector)):
    print("Dumping match %d" % (n))
    p.dump_pdb(path.join(outdir, "foo%d.pdb" % (n)))

print("DONE")

