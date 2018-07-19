import pyrosetta
import privileged_residues

import os

from datetime import date
from os import path
from privileged_residues.postproc import filter_clash_minimize

pr = privileged_residues.PrivilegedResidues()
pose = pyrosetta.pose_from_pdb("/home/onalant/source/privileged_residues/local/5L6HC3_asymmetric.pdb")
selector = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()

hits = pr.search(pose, ["bidentate"], selector)

outdir = path.expanduser("~/dump/%s_PRDemo/" % (date.today().isoformat()))
os.makedirs(outdir, exist_ok=True)

for (n, hit) in enumerate(filter_clash_minimize(pose, hits)):
	hit.dump_pdb(path.join(outdir, "match_%d.pdb" % (n)))

